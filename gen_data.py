import os
import torch
import argparse

from problems.utpp.problem_utpp import generate_utpp_instance
from problems.rtpp.problem_rtpp import generate_rtpp_instance

from utils.data_utils import check_extension, save_dataset

def generate_utpp_data(dataset_size, graph_size, num_products, max_price):
    return [generate_utpp_instance(graph_size, num_products, max_price)
            for i in range(dataset_size)]

def generate_rtpp_data(dataset_size, graph_size, num_products, max_price, max_supply, coeff):
    return [generate_rtpp_instance(graph_size, num_products, max_price, max_supply, coeff)
            for i in range(dataset_size)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None, help='Filename of the dataset to create')
    parser.add_argument('--data_dir', default='data/random', help='Create datasets in data_dir/problem')
    parser.add_argument('--problem', type=str, default='rtpp', help='problem')

    parser.add_argument('--dataset_size', type=int, default=10, help='Size of the dataset')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')

    parser.add_argument('--graph_size', type=int, default=150, help='Graph size of problem instances')
    parser.add_argument('--num_products', type=int, default=150, help='Number of products to be purchased')
    parser.add_argument('--max_supply', type=int, default=15, help='The max quantity in all markets')
    parser.add_argument('--max_price', type=int, default=10, help='Max price in all markets')
    parser.add_argument('--coeff', type=float, default=0.99, help='Demand coefficient in RTPP')

    opts = parser.parse_args()

    datadir = opts.data_dir
    os.makedirs(datadir, exist_ok=True)
    if opts.problem == 'utpp':
        filename = opts.filename if opts.filename is not None\
            else '{}_{:d}_{:d}.pkl'.format(opts.problem, opts.graph_size, opts.num_products)
    elif opts.problem == 'rtpp':
        filename = opts.filename if opts.filename is not None \
            else '{}_{:d}_{:d}_{}.pkl'.format(opts.problem, opts.graph_size, opts.num_products, str(opts.coeff))
    else:
        filename = opts.filename
    filename = os.path.join(datadir, filename)

    torch.manual_seed(opts.seed)
    if opts.problem == 'utpp':
        dataset = generate_utpp_data(opts.dataset_size, opts.graph_size, opts.num_products, opts.max_price)
    elif opts.problem == 'rtpp':
        dataset = generate_rtpp_data(opts.dataset_size, opts.graph_size, opts.num_products, opts.max_price,
                                     opts.max_supply, opts.coeff)
    else:
        dataset = None
        print('Currently unsupported problem: {}!'.format(opts.problem))

    save_dataset(dataset, filename)