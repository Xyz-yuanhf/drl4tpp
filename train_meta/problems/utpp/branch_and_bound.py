import numpy as np
import gurobipy as gp
from gurobipy import GRB
from os.path import join

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)
        # find the shortest cycle in the selected edge list
        to_sep, tour = subtour(x_val, y_val)
        if to_sep:
            # add subtour elimination constr.
            other_city = [city for city in range(num_city) if city not in tour]
            indices = [(i, j) for i in tour for j in other_city]
            for city in tour:
                model.cbLazy(gp.quicksum(model._x[i, j] for i, j in indices) >= model._y[city])


# Given a tuplelist of edges, find the shortest subtour

def subtour(x, y):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in x.keys()
                         if x[i, j] > 0.5)
    # print(edges)
    unvisited = []
    for city in range(num_city):
        if y[city] > 0.5:
            unvisited.append(city)
    cycle = range(num_city + 1)  # initial length has 1 more city
    num_cycle = 0
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for _, j in edges.select(current, '*')
                         if j in unvisited]
        # print(thiscycle)
        num_cycle += 1
        if num_cycle == 1 or 0 in cycle or len(cycle) > len(thiscycle):
            cycle = thiscycle
    return num_cycle > 1 or 0 not in cycle, cycle


def parse_data_STPP(data_fn):
    with open(data_fn, 'r') as f:
        lines = f.readlines()
    num_info_line = 6
    data_info = lines[:num_info_line]

    # coordinates
    num_city = int(data_info[3].split()[-1])
    coordinates = np.zeros((num_city, 2))
    coord_sec_start = num_info_line
    for i in range(num_city):
        coordinates[i] = [float(x) for x in lines[coord_sec_start + 1 + i].split()[1:]]

    # demand
    dmd_sec_start = coord_sec_start + num_city + 1
    num_prod = int(lines[dmd_sec_start + 1])
    demand = np.array([int(x.split()[-1])
                       for x in lines[dmd_sec_start + 2: dmd_sec_start + 2 + num_prod]])

    # suppliers
    offer_sec_start = dmd_sec_start + 2 + num_prod
    city_prods = [{}]
    for line in lines[offer_sec_start + 1 + 1: offer_sec_start + 1 + num_prod]:
        arr_num = line.split()
        num_offer = int(arr_num[1])
        products = {}
        for i in range(num_offer):
            prod_id = int(arr_num[2 + i * 3]) - 1
            prod_cost = float(arr_num[2 + i * 3 + 1])
            prod_cap = float(arr_num[2 + i * 3 + 2])
            products[prod_id] = (prod_cost, prod_cap)
        city_prods.append(products)

    # get available cities for each product
    prod_cities = [[] for _ in range(num_prod)]
    prod_prices = [[] for _ in range(num_prod)]
    for city in range(num_city):
        for prod_id in city_prods[city].keys():
            prod_cities[prod_id].append(city)
            prod_prices[prod_id].append(city_prods[city][prod_id][0])

    # get travel cost matrix
    dist = {(i, j):
                int(np.linalg.norm(coordinates[i] - coordinates[j]))
            for i in range(num_city) for j in range(num_city) if i != j}

    return num_city, num_prod, dist, demand, city_prods, prod_cities, prod_prices


################### get data ###################
path = '../../data/benchmark/Class_4'
data_fn = 'CapEuclideo.50.50.99.5.tpp'
data_fn = join(path, data_fn)
num_city, num_prod, dist, demand, city_prods, prod_cities, prod_prices = parse_data_STPP(data_fn)
# for i in range(num_prod):
#     d = demand[i]
#     if np.random.random() > 0.6:
#         d = np.random.randint(d, int(d * 1.3))
#         demand[i] = d

m = gp.Model()

# Create variables
x = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
y = m.addVars(num_city, vtype=GRB.BINARY, name='y')
z = []
for prod_id in range(num_prod):
    z.append(m.addVars(len(prod_cities[prod_id]),
                       obj=prod_prices[prod_id],
                       name='z_%d' % prod_id))

# demand constraints
m.addConstrs(gp.quicksum(z[prod_id][i] for i in range(len(prod_cities[prod_id])))
             >= demand[prod_id] for prod_id in range(num_prod))
# capacity constraints
for prod_id in range(num_prod):
    for i in range(len(prod_cities[prod_id])):
        city = prod_cities[prod_id][i]
        m.addConstr(z[prod_id][i] <= y[city] * city_prods[city][prod_id][1])
# indegree and outdegree constraints
m.addConstrs(x.sum('*', i) == y[i] for i in range(num_city))
m.addConstrs(x.sum(i, '*') == y[i] for i in range(num_city))

# Optimize model
m._x = x
m._y = y
m.Params.LazyConstraints = 1
m.Params.TimeLimit = 120
m.optimize(subtourelim)
# m.optimize()

x_val = m.getAttr('X', x)
y_val = m.getAttr('X', y)
_, tour = subtour(x_val, y_val)

print('')
print('Optimal tour: %s' % str(tour))
print('Num of city:  %d' % len(tour))
print('Optimal cost: %g' % m.ObjVal)
print('Runtime: %.2fs' % m.Runtime)
print('')

# # save the result
# line = []
# line.append("Obj: %.2f\n" % m.ObjVal)
# line.append("Path: " + str(tour) + "\n")
# line.append('Lazy_constraints runtime: %.2fs\n' % m.Runtime)
# with open('./optimal/tsp_%d.txt' % n, 'w') as f:
#     f.writelines(line)

