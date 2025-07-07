import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.data_utils import load_dataset
from train_meta.problems.utpp.state_utpp import StateUTPP

class UTPP(object):

    NAME = 'utpp'

    @staticmethod
    def get_costs(dataset, pi):
        batch_size = pi.size()[0]
        max_coord, max_price = 1000, 10  # norm scale
        ids = torch.arange(batch_size, dtype=torch.int64, device=pi.device)[:, None]

        # Travelling cost
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1) * max_coord
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        length = (
            torch.floor((d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1)).sum(1)  # Prevent error if len 1 seq
            + torch.floor((d[:, 0] - loc_with_depot[:, 0]).norm(p=2, dim=-1))  # Depot to first
            + torch.floor((d[:, -1] - loc_with_depot[:, 0]).norm(p=2, dim=-1))  # Last to depot, 0 if depot is last
        )
        # Purchasing cost
        price = (max_price + 1) * (1-dataset['supply_data']) + max_price * dataset['price_data']
        price = F.pad(price, (0, 0, 1, 0), value=max_price+1)
        price = price[ids, pi]
        price = torch.sum(torch.min(price, dim=1)[0], dim=-1)
        # Total cost
        cost_norm = 1000
        cost = (length + price) / cost_norm

        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return UTPPDataset(**kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateUTPP.initialize(*args, **kwargs)

def generate_utpp_instance(graph_size, num_products, max_price):
    loc = torch.randint(low=0, high=1001, size=[graph_size - 1, 2], dtype=torch.float) / 1000.
    depot = torch.randint(low=0, high=1001, size=[2], dtype=torch.float) / 1000.
    # number & price of product_j supplied at market_i
    supply_data = torch.zeros(graph_size - 1, num_products, dtype=torch.float)  # number of product_j supplied at market_i
    price_data = torch.zeros(graph_size - 1, num_products, dtype=torch.float)  # price of product_j supplied at market_i
    num_prod_market = torch.randint(1, graph_size - 1, size=[num_products])
    num_prod_market, _ = torch.sort(num_prod_market)
    indices_1 = torch.cat([torch.randperm(graph_size - 1)[:num_prod_market[i]] for i in range(num_products)])  # market
    indices_2 = torch.tensor([i for i in range(num_products) for j in range(num_prod_market[i])])  # product
    supply_data[(indices_1, indices_2)] = 1
    price_data[(indices_1, indices_2)] = torch.randint(low=1, high=max_price + 1,
                                                       size=[len(indices_1)],
                                                       dtype=torch.float)
    price_data = price_data / max_price

    coo_indices = torch.stack((indices_1, indices_2))
    supply_sparse = torch.sparse_coo_tensor(coo_indices, torch.ones_like(indices_1), [graph_size - 1, num_products])

    return {
        'loc': loc,
        'depot': depot,
        'supply_data': supply_data,
        'price_data': price_data,
        'supply_sparse': supply_sparse
    }

class UTPPDataset(Dataset):
    def __init__(self, filename=None, num_samples=1280, num_reuse=1, rand=True,
                 batch_size=512, problem_config=[], max_price=10):
        if filename is not None:
            self.data = load_dataset(filename)
        else:
            if not rand:
                self.data = []
                for config in problem_config:
                    graph_size, num_products = config['graph_size'], config['num_products']
                    for i in range(batch_size):
                        self.data.append(generate_utpp_instance(graph_size, num_products, max_price))
            else:
                if num_reuse == 1:
                    self.data = []
                    for i in range(num_samples):
                        if i % batch_size == 0:
                            config = random.choice(problem_config)
                            graph_size, num_products = config['graph_size'], config['num_products']
                        self.data.append(generate_utpp_instance(graph_size, num_products, max_price))
                else:
                    print(f'Generating {num_samples // num_reuse} instances...')
                    self.data = []
                    for i in tqdm(range(num_samples // num_reuse)):
                        if i % batch_size == 0:
                            config = random.choice(problem_config)
                            graph_size, num_products = config['graph_size'], config['num_products']
                        self.data.append(generate_utpp_instance(graph_size, num_products, max_price))
                    self.data = self.data * num_reuse
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

