import os
import torch
import argparse
import numpy as np
import pandas as pd

from train_single.train import get_inner_model
from utils import torch_load_cpu, load_problem, move_to
from evaluation.utils import augment_inference
from nets.attention_model import AttentionModel
import train_single.problems.rtpp.rtpp_baseline as bl_rtpp

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, help='Path to load model from previous checkpoint file')
parser.add_argument('--data_dir', default='data/benchmark/Class_4', help='Directory to load benchmark')
parser.add_argument('--problem', type=str, default='rtpp', help='problem')
parser.add_argument('--augment', type=bool, default=True, help='if use ×8 augmentation')

parser.add_argument('--graph_size', type=int, default=50, help='Graph size of problem instances')
parser.add_argument('--num_products', type=int, default=50, help='Number of products to be purchased')
parser.add_argument('--max_supply', type=int, default=15, help='The max quantity in all markets')
parser.add_argument('--max_price', type=int, default=10, help='Max price in all markets')
parser.add_argument('--coeff', type=float, default=0.99, help='Demand coefficient in RTPP')

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
    opts.model_path = 'ckpt/rtpp_{:d}_{:d}_{}/pretrain.pt'.format(opts.graph_size, opts.num_products, str(opts.coeff))
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

if (opts.graph_size, opts.num_products, opts.coeff) == (50, 50, 0.9):
    opt_cost = [3650, 3811, 3522, 3329, 3543]
elif (opts.graph_size, opts.num_products, opts.coeff) == (50, 50, 0.95):
    opt_cost = [2636, 1758, 2390, 2643, 2792]
elif (opts.graph_size, opts.num_products, opts.coeff) == (50, 50, 0.99):
    opt_cost = [1703, 1305, 1893, 2143, 2266]
elif (opts.graph_size, opts.num_products, opts.coeff) == (50, 100, 0.9):
    opt_cost = [4718, 4705, 4462, 4620, 4834]
elif (opts.graph_size, opts.num_products, opts.coeff) == (50, 100, 0.95):
    opt_cost = [3429, 3226, 2818, 3190, 3273]
elif (opts.graph_size, opts.num_products, opts.coeff) == (50, 100, 0.99):
    opt_cost = [2389, 1986, 2091, 2519, 2580]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 50, 0.9):
    opt_cost = [4847, 4672, 4381, 4861, 4607]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 50, 0.95):
    opt_cost = [2699, 2685, 2649, 3272, 2995]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 50, 0.99):
    opt_cost = [1313, 1367, 1146, 2217, 1475]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 100, 0.9):
    opt_cost = [6494, 6375, 6063, 6863, 6415]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 100, 0.95):
    opt_cost = [3484, 3468, 3491, 3532, 3802]
elif (opts.graph_size, opts.num_products, opts.coeff) == (100, 100, 0.99):
    opt_cost = [1539, 1554, 1672, 2168, 2394]
elif (opts.graph_size, opts.num_products, opts.coeff) == (150, 150, 0.9):
    opt_cost = [9518, 9478, 9456, 9609, 9398]
elif (opts.graph_size, opts.num_products, opts.coeff) == (150, 150, 0.95):
    opt_cost = [4789, 5099, 5037, 5271, 5662]
elif (opts.graph_size, opts.num_products, opts.coeff) == (150, 150, 0.99):
    opt_cost = [1702, 2547, 2596, 2202, 2828]
else:
    raise RuntimeError("Optimal obj. not imported")

rl_cost_ls, post_cost_ls = [], []
rl_gap_ls, post_gap_ls = [], []


for id in range(0, 5):
    # read data
    instance_name = 'CapEuclideo.{:d}.{:d}.{}.{:d}.tpp'. format(opts.graph_size, opts.num_products,
                                                                str(opts.coeff)[2:], id+1)

    print('{}:'.format(instance_name))
    print('Optimal: {:d}'.format(opt_cost[id]))
    file_path = os.path.join(opts.data_dir, instance_name)
    num_markets, num_products, x_coord, y_coord, demand, supply_data, price_data = bl_rtpp.read_data(file_path)
    dist_matrix = bl_rtpp.get_dist_matrix(x_coord, y_coord)

    # data transformation for TRH
    instance = {}
    loc_with_depot = torch.stack((torch.tensor(x_coord, dtype=torch.float),
                                  torch.tensor(y_coord, dtype=torch.float)), dim=0).t()
    loc_with_depot = loc_with_depot / 1000.
    instance['loc'] = loc_with_depot[1:, :].unsqueeze(0)
    instance['depot'] = loc_with_depot[0, :].unsqueeze(0)
    instance['supply_data'] = torch.tensor((supply_data / demand)[1:], dtype=torch.float).unsqueeze(0)
    instance['price_data'] = torch.tensor((price_data * demand)[1:], dtype=torch.float).unsqueeze(0)
    instance['price_data'] = instance['price_data'] / (opts.max_price * opts.max_supply)
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
    tour = np.array([0] + [i for i in tour if i != 0])
    _, new_tour = bl_rtpp.routing_construction(tour, x_coord, y_coord)  # tsp re-opt
    tour = tour[new_tour]
    tour = bl_rtpp.TRH_for_RTPP(tour.tolist(), dist_matrix, demand, supply_data, price_data)  # TRH
    length, new_tour = bl_rtpp.routing_construction(tour[: -1], x_coord, y_coord)
    purchase, _ = bl_rtpp.product_purchase_planning(tour, num_products, demand, supply_data, price_data)
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
df.index = ['CE.{:d}.{:d}.{:d}.tpp'. format(opts.graph_size, opts.num_products, id+1)
            for id in range(0, 5)]
df.loc['Average'] = df.mean()

for i, col in enumerate(df.columns):
    if i % 2 == 0:
        df[col] = df[col].apply(lambda x: round(x))
    else:
        df[col] = df[col].apply(lambda x: format(x, '.2%'))

display_cols = df.columns[:4]
print(df[display_cols])