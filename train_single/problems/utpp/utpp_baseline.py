import os
import random
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# suppress the output log from concorde
class suppress_stdout_stderr(object):
    '''
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

# read data file
def read_data(file_path):
    f = open(file_path)
    [f.readline() for j in range(3)]
    num_markets = int(f.readline().split()[-1])
    [f.readline() for j in range(3)]
    # market coordinate
    x_coord, y_coord = [], []
    for i in range(num_markets):
        line = f.readline().split()
        x, y = int(line[1]), int(line[2])
        x_coord.append(x)
        y_coord.append(y)
    x_coord, y_coord = np.array(x_coord), np.array(y_coord)
    # product demand
    [f.readline() for j in range(1)]
    num_products = int(f.readline().split()[-1])
    demand = []
    for i in range(num_products):
        line = f.readline().split()
        d = int(line[1])
        demand.append(d)
    demand = np.array(demand)
    # number & price of product_j supplied at market_i
    [f.readline() for j in range(1)]
    supply_data = np.zeros((num_markets, num_products))  # number of product_j supplied at market_i
    price_data = np.zeros((num_markets, num_products))  # price of product_j supplied at market_i
    for i in range(num_markets):
        line = f.readline().split()
        market_id = int(line[0]) - 1
        num_prod_market = int(line[1])
        for j in range(num_prod_market):
            product_id = int(line[3 * j + 2]) - 1
            price, supply = int(line[3 * j + 3]), int(line[3 * j + 4])
            price_data[market_id, product_id] = price
            supply_data[market_id, product_id] = supply
    # resort supply_data & price_data & demand
    order = np.argsort(np.sum(supply_data, axis=0))
    supply_data = supply_data[:, order]
    price_data = price_data[:, order]
    demand = demand[order]
    return num_markets, num_products, x_coord, y_coord, demand, supply_data, price_data

# get distance matrix
def get_dist_matrix(x_coord, y_coord):
    num_nodes = len(x_coord)
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(0, num_nodes):
        for j in range(i + 1, num_nodes):
            coord_i = np.array([x_coord[i], y_coord[i]])
            coord_j = np.array([x_coord[j], y_coord[j]])
            dist = int(np.linalg.norm(coord_i - coord_j))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# ortools TSP solver
def tsp_solver(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # get route (tour)
    index = routing.Start(0)
    tour = [manager.IndexToNode(0)]
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        tour.append(manager.IndexToNode(index))
    tour = np.array(tour)

    return solution.ObjectiveValue(), tour

# cheapest insertion construction
def cheapest_insertion(tour, target_market, dist_matrix):
    h = target_market
    lowest_increase, lowest_position = float('inf'), None
    for index in range(len(tour) - 1):
        i, j = tour[index], tour[index + 1]
        routing_cost_increase = dist_matrix[i, h] + dist_matrix[h, j] - dist_matrix[i, j]
        if routing_cost_increase < lowest_increase:
            lowest_increase = routing_cost_increase
            lowest_position = (index, index + 1)
    return lowest_increase, lowest_position

# TSP re-optimization
def routing_construction(selected_markets, x_coord, y_coord):
    len_padding = 5 - len(selected_markets) if len(selected_markets) <= 4 else False
    if len_padding > 0:
        selected_markets = np.concatenate((selected_markets, np.array([0 for i in range(len_padding)])))
    x_coord = x_coord[selected_markets]
    y_coord = y_coord[selected_markets]
    dist_matrix = get_dist_matrix(x_coord, y_coord)
    obj, tour = tsp_solver(dist_matrix)
    if len_padding > 0:
        tour = np.array([i for i in tour if i < 5 - len_padding])
    return obj, tour

# get purchase plan
def product_purchase_planning(selected_markets, num_products, demand, supply_data, price_data):
    purchasing_cost = 0
    purchasing_plan = np.zeros_like(supply_data, dtype=int)
    supply_data = supply_data[selected_markets]
    price_data = price_data[selected_markets]
    num_purchased_ls = []
    for product_id in range(num_products):
        demand_quantity = demand[product_id]
        price_info = price_data[:, product_id] + 1e6 * (1 - supply_data[:, product_id])
        purchase_priority = np.argsort(price_info)
        # find the market with lower price
        num_purchased = 0
        for market_id in purchase_priority:
            quantity = min(supply_data[market_id, product_id], demand_quantity - num_purchased)
            price = price_data[market_id, product_id]
            num_purchased += quantity
            purchasing_cost += price * quantity
            purchasing_plan[market_id, product_id] = quantity
            if num_purchased >= demand_quantity:
                break
        num_purchased_ls.append(num_purchased)
    not_purchased = demand - np.array(num_purchased_ls)
    if not_purchased.any() > 0.5:
        # print('Unfeasible solution!')
        purchasing_cost = 1e6
    return purchasing_cost, purchasing_plan

# Generalized Savings Heuristic for U-TPP
def GSH_for_UTPP(num_markets, dist_matrix, supply_data, price_data):
    # find the market which sells more products than any other market at the cheapest price
    price_data = price_data + 1e6 * (1 - supply_data)
    min_price = np.min(price_data, axis=0)  # min price of each product among all markets
    min_price = np.tile(np.atleast_2d(min_price), (num_markets, 1))  # (num_markets, num_products)
    num_cheapest = np.sum(np.equal(price_data, min_price), axis=1)  # num of cheapest products in each markets
    max_cheapest, market_index = 0, []
    for i, n in enumerate(num_cheapest):
        if n > max_cheapest:
            max_cheapest = n
            market_index = [i]
        elif n == max_cheapest:
            market_index.append(i)
    # resolve ties
    selected_index = market_index[0] if len(market_index) == 1 \
        else market_index[np.argmin(np.sum(price_data[market_index, :], axis=1))]
    current_tour = [0, selected_index, 0]  # initial tour
    unvisited_markets = [i for i in range(num_markets) if i not in current_tour]

    while True:
        # compute f(x, l) & g(x, p, l) -> savings
        saving_ls = []
        f_product = np.min(price_data[current_tour[:-1], :], axis=0)
        for market_id in unvisited_markets:
            # decrease in purchasing cost
            g_product_market = np.maximum(f_product - price_data[market_id], 0)
            purchasing_cost_decrease = np.sum(g_product_market)
            # increase in routing cost
            # 检查到这里了
            routing_cost_increase, insert_position = cheapest_insertion(current_tour, market_id, dist_matrix)
            # total_savings
            saving = purchasing_cost_decrease - routing_cost_increase
            if saving > 0:
                saving_ls.append((market_id, saving, insert_position))  # market_id, saving, insert_position
        if len(saving_ls) == 0:
            break
        market_id, saving, insert_pos = max(saving_ls, key=lambda x: x[1])
        # update current_tour & unvisited_markets
        current_tour.insert(insert_pos[1], market_id)
        unvisited_markets.remove(market_id)
    return current_tour

# Tour Reduction Heuristic for U-TPP
def TRH_for_UTPP(init_markets, dist_matrix, supply_data, price_data):
    price_data = price_data + 1e6 * (1 - supply_data)
    current_tour = init_markets
    while True:
        # compute savings
        saving_ls = []
        f_product = np.min(price_data[current_tour[:-1], :], axis=0)
        for index, market_id in enumerate(current_tour):
            if market_id == 0:
                continue
            i, j, h = current_tour[index-1], current_tour[index+1], current_tour[index]
            excluded_tour = current_tour[: index] + current_tour[index+1:]
            # decrease in routing cost
            routing_cost_decrease = dist_matrix[i, h] + dist_matrix[h, j] - dist_matrix[i, j]
            # increase in purchasing cost
            # product potential to make an increase on purchasing cost
            min_position = np.where(np.equal(f_product, price_data[market_id, :]))[0]
            f_product_current = f_product[min_position]
            f_product_excluded = np.min(price_data[excluded_tour[:-1], :][:, min_position], axis=0)
            # product potential to make an increase on purchasing cost
            purchasing_cost_increase = np.sum(f_product_excluded - f_product_current)
            saving = routing_cost_decrease - purchasing_cost_increase
            if saving > 0:
                saving_ls.append((market_id, saving, index))  # market_id, saving, index
        if len(saving_ls) == 0:
            break
        market_id, saving, index = max(saving_ls, key=lambda x: x[1])
        # update current_tour & unvisited_markets
        del(current_tour[index])
    return current_tour

# Commodity Adding Heuristic for U-TPP
def CAH_for_UTPP(num_markets, dist_matrix, supply_data, price_data):
    price_data = price_data + 1e6 * (1 - supply_data)
    num_products = price_data.shape[1]

    random.seed(0)
    order = list(range(num_products))
    random.shuffle(order)
    # order.reverse()

    current_tour = [0, 0]  # initial tour
    unvisited_markets = [i for i in range(num_markets) if i not in current_tour]

    for p in order:
        saving_ls = []
        f_product = np.min(price_data[current_tour[:-1], p], axis=0)
        for market_id in unvisited_markets:
            # decrease in purchasing cost
            g_product_market = np.maximum(f_product - price_data[market_id, p], 0)
            purchasing_cost_decrease = np.sum(g_product_market)
            # increase in routing cost
            routing_cost_increase, insert_position = cheapest_insertion(current_tour, market_id, dist_matrix)
            # total_savings
            saving = purchasing_cost_decrease - routing_cost_increase
            if saving > 0:
                saving_ls.append((market_id, saving, insert_position))  # market_id, saving, insert_position
        if len(saving_ls) == 0:
            continue
        market_id, saving, insert_pos = max(saving_ls, key=lambda x: x[1])
        # update current_tour & unvisited_markets
        current_tour.insert(insert_pos[1], market_id)
        unvisited_markets.remove(market_id)
    return current_tour
