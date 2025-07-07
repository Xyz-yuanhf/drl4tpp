import torch
import argparse
from torch.utils.data import DataLoader

from nets.attention_model import AttentionModel
from train_single.train import get_inner_model
from utils.data_utils import load_dataset
from evaluation.utils import augment_inference
from utils import torch_load_cpu, load_problem, move_to
import train_single.problems.rtpp.rtpp_baseline as bl_rtpp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, help='Path to load model from previous checkpoint file')
    parser.add_argument('--data_path', default=None, help='Path to load evaluation data')
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
        opts.model_path = 'ckpt/rtpp_{:d}_{:d}_{}/pretrain.pt'.format(opts.graph_size, opts.num_products,
                                                                            str(opts.coeff))
    if opts.data_path is None:
        opts.data_path = 'data/random/rtpp_{:d}_{:d}_{}.pkl'.format(opts.graph_size, opts.num_products,
                                                                          str(opts.coeff))
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_data = torch_load_cpu(opts.model_path)
    val_dataset = load_dataset(opts.data_path)

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

    # Dataloader
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    rl_cost_ls, post_cost_ls, = [], []
    for id, instance_dataloader in enumerate(dataloader):
        instance_dataset = val_dataset[id]
        # data transformation for TRH
        xy_coord = 1000 * torch.cat((instance_dataloader['depot'],
                                     instance_dataloader['loc'].squeeze()), dim=0).numpy()
        x_coord, y_coord = xy_coord[:, 0], xy_coord[:, 1]
        dist_matrix = bl_rtpp.get_dist_matrix(x_coord, y_coord)
        demand = instance_dataloader['demand'].squeeze().numpy()
        padding = torch.zeros_like(instance_dataloader['supply_data'].squeeze()[0:1, :])
        supply_data = torch.cat((padding, instance_dataloader['supply_data'].squeeze()), dim=0).numpy() * demand
        price_data = torch.cat((padding, instance_dataloader['price_data'].squeeze()), dim=0).numpy() \
                     * opts.max_price * opts.max_supply / demand

        # model.forward()
        instance_dataloader = augment_inference(instance_dataloader) if opts.augment else instance_dataloader
        cost, _, pi = model(move_to(instance_dataloader, opts.device), return_pi=True)
        min_id = torch.argmin(cost)
        cost, pi = cost[min_id], pi[min_id]
        rl_cost = cost.item() * 1000  # rl_cost
        rl_cost_ls.append(rl_cost)
        # post-optimize
        tour = pi.cpu().squeeze().numpy()
        tour[1:], tour[0] = tour[:-1], 0
        _, new_tour = bl_rtpp.routing_construction(tour, x_coord, y_coord)  # tsp re-opt
        tour = tour[new_tour]
        tour = bl_rtpp.TRH_for_RTPP(tour.tolist(), dist_matrix, demand, supply_data, price_data)  # TRH
        length, new_tour = bl_rtpp.routing_construction(tour[: -1], x_coord, y_coord)
        purchase, _ = bl_rtpp.product_purchase_planning(tour, opts.num_products,
                                                        demand, supply_data, price_data)
        post_cost = length + purchase  # post_cost
        post_cost_ls.append(post_cost)

        # print
        print('Instance: {:d}/{:d}'.format(id + 1, len(dataloader)))
        print('RL-E2E: {:d},'.format(round(rl_cost)), 'RL+TRH: {:d} \n'.format(round(post_cost)))

    print(f'Model path: {opts.model_path}')
    print(f'Data path:  {opts.data_path}')
    print()
    print('------Average------')
    print('RL-E2E Obj.: {:d}'.format(round(sum(rl_cost_ls) / len(rl_cost_ls))))
    print('RL+TRH Obj.: {:d}'.format(round(sum(post_cost_ls) / len(post_cost_ls))))
    print('-------------------')
