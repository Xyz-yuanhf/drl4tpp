import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import load_dataset
from train_single.problems.rtpp.state_rtpp import StateRTPP

class RTPP(object):

    NAME = 'rtpp'

    @staticmethod
    def get_costs(dataset, pi):
        batch_size = pi.size()[0]
        max_coord, max_price, max_supply = 1000, 10, 15  # norm scale
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
        higher_price = torch.max(dataset['price_data']) + 1
        price_masked = higher_price * (dataset['supply_data'] == 0) + dataset['price_data']
        price_masked = F.pad(price_masked, (0, 0, 1, 0), value=higher_price)
        price_masked = price_masked[ids, pi]
        price, price_indices = torch.sort(price_masked, dim=1)  # sorted price
        price = price * max_price * max_supply

        supply = F.pad(dataset['supply_data'], (0, 0, 1, 0), value=0.)
        supply = supply[ids, pi]
        supply = torch.gather(supply, dim=1, index=price_indices)  # as order of sorted price
        demand_remaining = torch.ones(price.size(0), price.size(2), device=price.device)
        purchase_cost = torch.zeros(price.size(0), price.size(2), device=price.device)
        for i in range(price.size(1)):  # num_markets
            cur_supply = supply[:, i, :]  # market i
            cur_price = price[:, i, :]
            purchase = torch.minimum(demand_remaining, cur_supply)
            purchase_cost += (purchase * cur_price)
            demand_remaining = demand_remaining - purchase
            if torch.max(demand_remaining) == 0:
                break
        purchase_cost = torch.sum(purchase_cost, dim=-1)
        # Total cost
        cost_norm = 1000
        cost = (length + purchase_cost) / cost_norm
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return RTPPDataset(**kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateRTPP.initialize(*args, **kwargs)

# based on observation to the benchmark instances
def generate_rtpp_instance(graph_size, num_products, max_price, max_supply, coeff):
    loc = torch.randint(low=0, high=1001, size=[graph_size - 1, 2], dtype=torch.float) / 1000.
    depot = torch.randint(low=0, high=1001, size=[2], dtype=torch.float) / 1000.
    # number & price of product_j supplied at market_i
    supply_data = torch.zeros(graph_size - 1, num_products, dtype=torch.float)  # number of product_j supplied at market_i
    price_data = torch.zeros(graph_size - 1, num_products, dtype=torch.float)  # price of product_j supplied at market_i
    num_prod_market = torch.randint(1, graph_size - 1, size=[num_products])
    num_prod_market, _ = torch.sort(num_prod_market)
    indices_1 = torch.cat([torch.randperm(graph_size - 1)[:num_prod_market[i]] for i in range(num_products)])
    indices_2 = torch.tensor([i for i in range(num_products) for j in range(num_prod_market[i])])
    supply_data[(indices_1, indices_2)] = torch.randint(low=1, high=max_supply + 1,
                                                       size=[len(indices_1)],
                                                       dtype=torch.float)
    price_data[(indices_1, indices_2)] = torch.randint(low=0, high=max_price + 1,
                                                       size=[len(indices_1)],
                                                       dtype=torch.float)
    # demand
    max_product_supply, _ = torch.max(supply_data, dim=0)
    sum_product_supply = torch.sum(supply_data, dim=0)
    epsilon = 1e-8 if coeff == 0.9 else 0
    scale = (1 - coeff) / (1 - 0.1)
    basic_demand = torch.maximum(torch.floor(0.1 * max_product_supply), torch.ones(num_products)) + \
                   torch.floor(0.9 * sum_product_supply - epsilon)
    demand = torch.ceil(basic_demand * scale)

    # normalization
    supply_data = supply_data / demand
    price_data = price_data * demand / (max_price * max_supply)

    coo_indices = torch.stack((indices_1, indices_2))
    supply_sparse = torch.sparse_coo_tensor(coo_indices, torch.ones_like(indices_1), [graph_size - 1, num_products])

    return {
        'loc': loc,
        'depot': depot,
        'supply_data': supply_data,
        'price_data': price_data,
        'supply_sparse': supply_sparse,
        'demand': demand  # used in heuristic eval
    }


class RTPPDataset(Dataset):

    def __init__(self, filename=None, num_samples=1280, num_reuse=1,
                 graph_size=50, num_products=50, max_price=10, max_supply=15, coeff=0.9):
        if filename is not None:
            self.data = load_dataset(filename)
        else:
            print(f'Generating {num_samples // num_reuse} instances...')
            if num_reuse == 1:
                self.data = [
                    generate_rtpp_instance(graph_size, num_products, max_price, max_supply, coeff)
                    for i in tqdm(range(num_samples))
                ]
            else:
                self.data = [
                    generate_rtpp_instance(graph_size, num_products, max_price, max_supply, coeff)
                    for i in tqdm(range(num_samples // num_reuse))
                ] * num_reuse
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    torch.manual_seed(1234)
    dataset = RTPPDataset(num_samples=2, graph_size=20, num_products=50)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        RTPP.get_costs(dataset=batch, pi=None)
