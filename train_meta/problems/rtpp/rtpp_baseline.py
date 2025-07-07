import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

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

# or-tools TSP solver
def tsp_solver(dist_matrix):
    # Instantiate the data problem
    data = {'distance_matrix': dist_matrix,
            'num_vehicles': 1, 'depot': 0}
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    # Define cost of each arc
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index:
        data['distance_matrix']
        [manager.IndexToNode(from_index)]
        [manager.IndexToNode(to_index)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    # Get the tour
    index = routing.Start(0)
    tour = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        tour.append(manager.IndexToNode(index))

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
    x_coord = x_coord[selected_markets]
    y_coord = y_coord[selected_markets]
    dist_matrix = get_dist_matrix(x_coord, y_coord)
    obj, tour = tsp_solver(dist_matrix)
    return obj, tour

# get purchase plan
def product_purchase_planning(selected_markets, num_products, demand, supply_data, price_data):
    purchasing_cost = 0
    supply_data = supply_data[selected_markets]
    price_data = price_data[selected_markets]
    price_data = price_data + 1e6 * (1 - (supply_data > 0))
    price, price_indices = np.sort(price_data, axis=0), np.argsort(price_data, axis=0)
    for p in range(num_products):
        remain_demand_p = demand[p]
        for m in range(price.shape[0]):
            supply_quantity = supply_data[price_indices[m, p], p]
            purchase_quantity = min(remain_demand_p, supply_quantity)
            remain_demand_p = remain_demand_p - purchase_quantity
            purchasing_cost += purchase_quantity * price[m, p]
            if remain_demand_p == 0:
                break
            if supply_quantity == 0:
                print('Infeasible! Demand cannot be satisfied!')
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