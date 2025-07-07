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
    supply_data = supply_data[selected_markets]
    price_data = price_data[selected_markets]
    price_data = price_data + 1e6 * (1 - (supply_data > 0))
    price, price_indices = np.sort(price_data, axis=0), np.argsort(price_data, axis=0)
    total_remain_demand = 0
    for p in range(num_products):
        remain_demand_p = demand[p]
        for m in range(price.shape[0]):
            supply_quantity = supply_data[price_indices[m, p], p]
            purchase_quantity = min(remain_demand_p, supply_quantity)
            remain_demand_p = remain_demand_p - purchase_quantity
            purchasing_cost += purchase_quantity * price[m, p]
            if remain_demand_p == 0 or supply_quantity == 0:
                break
        total_remain_demand += remain_demand_p
    purchasing_cost += total_remain_demand * 10000
    return purchasing_cost, None

# Market Adding Heuristic for R-TPP
def MAH_for_RTPP(num_markets, dist_matrix, demand, supply_data, price_data):
    # find in which market each such product is available at the lowest price
    # and first inserts in the tour the market corresponding to the highest price
    price_data = price_data + 1e6 * (1 - (supply_data > 0))
    current_tour = [0, 0]
    unvisited_markets = [i for i in range(num_markets) if i not in current_tour]

    while True:
        # get unsatisfied product
        remaining_demand = demand - np.sum(supply_data[current_tour], axis=0)
        unsatisfied_product = np.where(remaining_demand > 0)[0]
        if len(unsatisfied_product) == 0:
            break
        un_price = price_data[unvisited_markets, :][:, unsatisfied_product]
        min_price = np.min(un_price, axis=0)
        index_min_price = np.argmin(un_price, axis=0)  # market_id with min price
        index_max_min_price = np.argmax(min_price)  # product_id with max-min price
        market_id = unvisited_markets[index_min_price[index_max_min_price]]
        # cheapest insertion
        _, insert_position = cheapest_insertion(current_tour, market_id, dist_matrix)
        current_tour.insert(insert_position[1], market_id)
        unvisited_markets.remove(market_id)
    return current_tour

# Tour Reduction Heuristic for R-TPP
def TRH_for_RTPP(init_markets, dist_matrix, demand, supply_data, price_data):
    current_tour = init_markets
    while True:
        purchasing_cost_0, _ = product_purchase_planning(current_tour, supply_data.shape[1],
                                                         demand, supply_data, price_data)
        possible_del = []
        for market_id in current_tour[1: -1]:
            infeasible = (np.sum(supply_data[init_markets], axis=0) - supply_data[market_id] - demand) < 0
            if np.sum(infeasible) == 0:
                possible_del.append(market_id)
        saving_ls = []
        for market_id in possible_del:
            index = current_tour.index(market_id)
            i, j, h = current_tour[index - 1], current_tour[index + 1], current_tour[index]
            excluded_tour = current_tour[: index] + current_tour[index + 1:]
            # decrease in routing cost
            routing_cost_decrease = dist_matrix[i, h] + dist_matrix[h, j] - dist_matrix[i, j]
            # increase in purchasing cost
            purchasing_cost_1, _ = product_purchase_planning(excluded_tour, supply_data.shape[1],
                                                             demand, supply_data, price_data)
            purchasing_cost_increase = purchasing_cost_1 - purchasing_cost_0
            saving = routing_cost_decrease - purchasing_cost_increase
            if saving > 0:
                saving_ls.append((market_id, saving))
        if len(possible_del) == 0 or len(saving_ls) == 0:
            break
        market_id, saving = max(saving_ls, key=lambda x: x[1])
        current_tour.remove(market_id)
    return current_tour

# Commodity Adding Heuristic for R-TPP
def CAH_for_RTPP(num_markets, dist_matrix, demand, supply_data, price_data):
    num_products = price_data.shape[1]

    # random.seed(0)
    order = list(range(num_products))
    random.seed(0)
    random.shuffle(order)

    # First units of product 1
    p_0 = order[0]
    unit_purchase_cost_ls = []
    for m in range(num_markets):
        if supply_data[m, p_0] > 0:
            unit_purchase_cost = 2 * dist_matrix[0, m] / supply_data[m, p_0] + price_data[m, p_0]
            unit_purchase_cost_ls.append((m, unit_purchase_cost))
    first_market_id = min(unit_purchase_cost_ls, key=lambda x: x[1])[0]

    current_tour = [0, first_market_id, 0]  # initial tour
    unvisited_markets = [i for i in range(num_markets) if i not in current_tour]

    for p in order:
        # purchase more unit
        while np.sum(supply_data[current_tour, p]) < demand[p]:  # product p not satisfied
            total_cost_ls = []
            for m in unvisited_markets:
                if supply_data[m, p] > 0:
                    routing_cost_increase, insert_position = cheapest_insertion(current_tour, m, dist_matrix)
                    total_purchase_cost, _ = product_purchase_planning(current_tour + [m], p + 1, demand[:p + 1],
                                                                       supply_data[:, :p + 1], price_data[:, :p + 1])
                    total_cost_ls.append((m, routing_cost_increase + total_purchase_cost, insert_position))
            market_id, total_cost, insert_pos = min(total_cost_ls, key=lambda x: x[1])
            # update current_tour & unvisited_markets
            current_tour.insert(insert_pos[1], market_id)
            unvisited_markets.remove(market_id)
        # purchase at lower price
        while True:
            current_purchase_cost, _ = product_purchase_planning(current_tour, p + 1, demand[:p + 1],
                                                               supply_data[:, :p + 1], price_data[:, :p + 1])
            saving_ls = []
            for m in unvisited_markets:
                if supply_data[m, p] > 0:
                    routing_cost_increase, insert_position = cheapest_insertion(current_tour, m, dist_matrix)
                    new_purchase_cost, _ = product_purchase_planning(current_tour + [m], p + 1, demand[:p + 1],
                                                                       supply_data[:, :p + 1], price_data[:, :p + 1])
                    saving = new_purchase_cost - current_purchase_cost + routing_cost_increase
                    if saving < 0:
                        saving_ls.append((m, saving, insert_position))
            if len(saving_ls) == 0:
                break
            else:
                market_id, saving, insert_pos = min(saving_ls, key=lambda x: x[1])
                # update current_tour & unvisited_markets
                current_tour.insert(insert_pos[1], market_id)
                unvisited_markets.remove(market_id)
    return current_tour

# Generalized Savings Heuristic for R-TPP
def GSH_for_RTPP(num_markets, dist_matrix, demand, supply_data, price_data):
    num_products = price_data.shape[1]

    order = list(range(num_products))
    random.seed(0)
    random.shuffle(order)

    # First units of product 1
    p_0 = order[0]
    unit_purchase_cost_ls = []
    for m in range(num_markets):
        if supply_data[m, p_0] > 0:
            unit_purchase_cost = 2 * dist_matrix[0, m] / supply_data[m, p_0] + price_data[m, p_0]
            unit_purchase_cost_ls.append((m, unit_purchase_cost))
    first_market_id = min(unit_purchase_cost_ls, key=lambda x: x[1])[0]

    current_tour = [0, first_market_id, 0]  # initial tour
    unvisited_markets = [i for i in range(num_markets) if i not in current_tour]

    for p in order:
        # purchase more unit
        while np.sum(supply_data[current_tour, p]) < demand[p]:  # product p not satisfied
            total_cost_ls = []
            for m in unvisited_markets:
                if supply_data[m, p] > 0:
                    routing_cost_increase, insert_position = cheapest_insertion(current_tour, m, dist_matrix)
                    total_purchase_cost, _ = product_purchase_planning(current_tour + [m], p + 1, demand[:p + 1],
                                                                       supply_data[:, :p + 1], price_data[:, :p + 1])
                    total_cost_ls.append((m, routing_cost_increase + total_purchase_cost, insert_position))
            market_id, total_cost, insert_pos = min(total_cost_ls, key=lambda x: x[1])
            # update current_tour & unvisited_markets
            current_tour.insert(insert_pos[1], market_id)
            unvisited_markets.remove(market_id)
        # purchase at lower price
    while True:
        current_purchase_cost, _ = product_purchase_planning(current_tour, num_products, demand,
                                                             supply_data, price_data)
        saving_ls = []
        for m in unvisited_markets:
            routing_cost_increase, insert_position = cheapest_insertion(current_tour, m, dist_matrix)
            new_purchase_cost, _ = product_purchase_planning(current_tour + [m], num_products, demand,
                                                             supply_data, price_data)
            saving = new_purchase_cost - current_purchase_cost + routing_cost_increase
            if saving < 0:
                saving_ls.append((m, saving, insert_position))
        if len(saving_ls) == 0:
            break
        else:
            market_id, saving, insert_pos = min(saving_ls, key=lambda x: x[1])
            # update current_tour & unvisited_markets
            current_tour.insert(insert_pos[1], market_id)
            unvisited_markets.remove(market_id)
    return current_tour