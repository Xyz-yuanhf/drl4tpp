import os
import torch
import argparse
import pandas as pd

from train_single.train import get_inner_model
from utils import torch_load_cpu, load_problem, move_to
from evaluation.utils import augment_inference
from nets.attention_model import AttentionModel
import train_single.problems.utpp.utpp_baseline as bl_utpp

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, help='Path to load model from previous checkpoint file')
parser.add_argument('--data_dir', default='data/benchmark/Class_3', help='Directory to load benchmark')
parser.add_argument('--problem', type=str, default='utpp', help='problem')
parser.add_argument('--augment', type=bool, default=True, help='if use ×8 augmentation')

parser.add_argument('--graph_size', type=int, default=50, help='Graph size of problem instances')
parser.add_argument('--num_products', type=int, default=50, help='Number of products to be purchased')
parser.add_argument('--max_price', type=int, default=10, help='Max price in all markets')

parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
parser.add_argument('--n_encode_layers', type=int, default=3,
                    help='Number of layers in the encoder/critic network')
parser.add_argument('--tanh_clipping', type=float, default=10.,
                    help='Clip the parameters to within +- this value using tanh. '
                         'Set to 0 to not perform any clipping.')
parser.add_argument('--normalization', default='batch', help='Normalization type, batch (default) or instance')

opts = parser.parse_args()

if opts.model_path is None:
    opts.model_path = 'ckpt/utpp_{:d}_{:d}/pretrain.pt'.format(opts.graph_size, opts.num_products)
opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_data = torch_load_cpu(opts.model_path)

# Overwrite model parameters by parameters to load
problem = load_problem(opts.problem)
model = AttentionModel(
    opts.embedding_dim,
    opts.hidden_dim,
    problem,
    n_products=opts.num_products,
    n_encode_layers=opts.n_encode_layers,
    mask_inner=True,
    mask_logits=True,
    normalization=opts.normalization,
    tanh_clipping=opts.tanh_clipping,
    checkpoint_encoder=False,
    shrink_size=None
).to(opts.device)
model_ = get_inner_model(model)
model_.load_state_dict({**model_.state_dict(), **model_data.get('model', {})})
model.set_decode_type('greedy')
model.eval()

if (opts.graph_size, opts.num_products) == (50, 50):
    opt_cost = [1856, 1070, 1553, 1394, 1536]
elif (opts.graph_size, opts.num_products) == (50, 100):
    opt_cost = [2397, 2138, 1852, 3093, 2603]
elif (opts.graph_size, opts.num_products) == (100, 50):
    opt_cost = [1468,  971, 1623, 1718, 2494]
elif (opts.graph_size, opts.num_products) == (100, 100):
    opt_cost = [2121, 1906, 1822, 1649, 2925]
elif (opts.graph_size, opts.num_products) == (150, 150):
    opt_cost = [1669, 2526, 2456, 1761, 2355]
elif (opts.graph_size, opts.num_products) == (150, 200):
    opt_cost = [1760, 2312, 2594, 1889, 2472]
elif (opts.graph_size, opts.num_products) == (200, 150):
    opt_cost = [1730, 2745, 1861, 2460, 2079]
elif (opts.graph_size, opts.num_products) == (200, 200):
    opt_cost = [1736, 2352, 2505, 3314, 2427]
else:
    raise RuntimeError("Optimal obj. not imported")

rl_cost_ls, post_cost_ls = [], []
rl_gap_ls, post_gap_ls = [], []

for id in range(0, 5):

    # read data
    instance_name = 'EEuclideo.{:d}.{:d}.{:d}.tpp'. format(opts.graph_size, opts.num_products, id+1)
    print('{}:'.format(instance_name))
    print('Optimal: {:d}'.format(opt_cost[id]))
    file_path = os.path.join(opts.data_dir, instance_name)
    num_markets, num_products, x_coord, y_coord, demand, supply_data, price_data = bl_utpp.read_data(file_path)
    dist_matrix = bl_utpp.get_dist_matrix(x_coord, y_coord)

    # data transformation for TRH
    instance = {}
    loc_with_depot = torch.stack((torch.tensor(x_coord, dtype=torch.float),
                                  torch.tensor(y_coord, dtype=torch.float)), dim=0).t()
    loc_with_depot = loc_with_depot / 1000.
    instance['loc'] = loc_with_depot[1:, :].unsqueeze(0)
    instance['depot'] = loc_with_depot[0, :].unsqueeze(0)
    instance['supply_data'] = torch.tensor(supply_data[1:, :], dtype=torch.float).unsqueeze(0)
    instance['price_data'] = torch.tensor(price_data[1:, :], dtype=torch.float).unsqueeze(0) \
                             / opts.max_price
    instance['supply_sparse'] = instance['supply_data'].to_sparse()
    instance = augment_inference(instance) if opts.augment else instance

    # model.forward()
    cost, _, pi = model(move_to(instance, opts.device), return_pi=True)
    min_id = torch.argmin(cost)
    cost, pi = cost[min_id], pi[min_id]
    rl_cost = cost.item() * 1000  # rl_cost
    rl_gap = (rl_cost - opt_cost[id]) / opt_cost[id]
    rl_cost_ls.append(rl_cost)
    rl_gap_ls.append(rl_gap)
    print('RL-E2E:  {:d} ({:.2%})'.format(round(rl_cost), rl_gap))

    # post-optimize
    tour = pi.squeeze().cpu().numpy()
    tour[1:], tour[0] = tour[:-1], 0
    _, new_tour = bl_utpp.routing_construction(tour, x_coord, y_coord)  # tsp re-opt
    tour = tour[new_tour]
    tour = bl_utpp.TRH_for_UTPP(tour.tolist(), dist_matrix, supply_data, price_data)  # TRH
    length, new_tour = bl_utpp.routing_construction(tour[: -1], x_coord, y_coord)
    purchase, _ = bl_utpp.product_purchase_planning(tour, opts.num_products,
                                                    demand, supply_data, price_data)
    post_cost = length + purchase  # post_cost
    post_gap = (post_cost - opt_cost[id]) / opt_cost[id]
    post_cost_ls.append(post_cost)
    post_gap_ls.append(post_gap)
    print('RL+TRH:  {:d} ({:.2%})'.format(round(post_cost), post_gap))
    print()

data = {'RL-E2E Obj.': rl_cost_ls,
        'RL-E2E Gap': rl_gap_ls,
        'RL+TRH Obj.': post_cost_ls,
        'RL+TRH Gap': post_gap_ls}

df = pd.DataFrame(data)
df.index = ['EE.{:d}.{:d}.{:d}.tpp'. format(opts.graph_size, opts.num_products, id+1)
            for id in range(0, 5)]
df.loc['Average'] = df.mean()

for i, col in enumerate(df.columns):
    if i % 2 == 0:
        df[col] = df[col].apply(lambda x: round(x))
    else:
        df[col] = df[col].apply(lambda x: format(x, '.2%'))

display_cols = df.columns[:4]
print(df[display_cols])
