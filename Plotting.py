'''
Make basic plotting utils to visualize our data
'''

import pandas as pd
import os
from scripts.milp import ReadData
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import json

royal_blue = [0, 20/256, 82/256]

def ReadSolution(datadir, file):
    '''
    Read an .xslx solution file into a solution dictionary
    '''
    #Read data
    variables = ["q", "w", "u", "y", "m"]
    indices = ["p", "i", "j", "k"]
    xls = pd.ExcelFile(os.path.join(datadir, file))
    sol = {}
    for variable in variables:
        var_pd = pd.read_excel(xls, sheet_name=variable)

        sol[variable] = {}
        active_indices = []
        for index in indices:
            if index in var_pd.columns:
                active_indices.append(index)
        keys = var_pd[active_indices].values
        target = var_pd[variable + "_final"].values
        for i in range(len(target)):
            sol[variable][tuple(keys[i])] = target[i]

    return sol

def just_circle(ax, x, y, radius=0.05):
    c = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=2.5,
               edgecolor=royal_blue + [0.6], facecolor='none',
               path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))])
    ax.add_artist(c)

def text(ax, x, y, text):
    ax.text(x, y, text, zorder=100,
            ha='center', va='top', weight='bold', color=royal_blue,
            style='italic', fontfamily='monospace',
            path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))])

def code(ax, x, y, text):
    ax.text(x, y, text, zorder=100,
            ha='center', va='top', weight='normal', color='0.0',
            fontfamily='Courier New', fontsize='medium',
            path_effects=[withStroke(linewidth=7, foreground=(1, 1, 1, 1))])

def circle(ax, x, y, txt, cde, radius=0.05):
    just_circle(ax, x, y, radius=radius)
    text(ax, x, y-0.2, txt)
    code(ax, x, y-0.33, cde)

def square(ax, x, y, txt, cde, radius=0.05):
    r = Rectangle((x, y), width=radius, height=radius, facecolor='none', edgecolor=royal_blue, linewidth=2.5)
    ax.add_artist(r)
    text(ax, x, y-0.2, txt)
    code(ax, x, y-0.33, cde)

def polygon(ax, x, y, numVertices, txt, cde, radius=0.05):
    r = RegularPolygon((x, y), numVertices, facecolor='none', edgecolor=royal_blue, linewidth=2.5, radius = radius)
    ax.add_artist(r)
    text(ax, x, y-0.2, txt)
    code(ax, x, y-0.33, cde)

def plot_solution(data, sol):
    '''
    Plot a solution
    Currently using color for vehicles, think to change that
    '''
    #First plot the data
    fig = plt.figure(figsize=(8, 8), facecolor='1')
    marg = 0.1
    ax = fig.add_axes([marg, marg, 1 - 1.8 * marg, 1 - 1.8 * marg], aspect=1, facecolor='1')

    #Plot all the points
    #Scaling the size of marker according to window size could be a good idea
    types = ["F", "D", "S", "C"]
    markers = ["o", "^", "s", "h"]
    labels = ["Firms", "Depots", "Satellites", "Customers"]
    for type, marker, label in zip(types, markers, labels):
        ax.scatter(data["XY"][data[type]][:,0], data["XY"][data[type]][:,1], s = 30, marker = marker, label = label)

    #Create the routes
    vehicle_routes, vehicle_tours_x, vehicle_tours_y = get_vehicle_routes(data, sol)

    #Then plot the solution
    V = ["b", "g", "r", "c", "m", "k"]
    P = ["--", ":", "-.", "-"]
    for k in vehicle_tours_x.keys():
        ax.plot(vehicle_tours_x[k], vehicle_tours_y[k], V[k], label=str(k))
    plt.legend()

    return 0

def get_total_demand(data, sol):
    '''
    Get a table in the form
    table[location][product] = demand of product p in location
    '''
    table = {}
    for location in data["DEM"]:
        table[location] = {j: data["DEM"][location][j] for j in range(len(data["P"]))}

    return table

def get_travel_distance(data, sol, vehicle_routes):

    travel_distance_table = {}
    for k, route in vehicle_routes.items():

        travel_distance = 0
        if len(route) > 1:
            for s in range(1, len(route)):
                i = route[s - 1]
                j = route[s]
                travel_distance += data["r"][i][j]

        travel_distance_table[k] = travel_distance

    return travel_distance_table

def get_travel_segments(data, sol, vehicle_routes):

    travel_segment_table = {}
    for k, route in vehicle_routes.items():

        travel_segment = []
        if len(route) > 1:
            for s in range(1, len(route)):
                i = route[s - 1]
                j = route[s]
                travel_segment.append(data["r"][i][j])

        travel_segment_table[k] = travel_segment

    return travel_segment_table

def get_average_load(data, sol, travel_segments, load_use):

    average_load_table = {}
    for k in travel_segments.keys():

        average = 0
        n_segments = len(travel_segments[k])
        total_distance = 0
        for i in range(n_segments):
            average += load_use[k][i] * travel_segments[k][i]
            total_distance += travel_segments[k][i]
        average /= total_distance
        average_load_table[k] = average

    return average_load_table

def plot_vehicle_load(data, sol, travel_segments, load_use, k, ax):

    x = []
    y = []
    current_time = 0
    current_load = 0
    max_load = data["Theta"][k]
    for i in range(len(travel_segments[k])):
        current_load = load_use[k][i]
        x.append(current_time)
        y.append(current_load / max_load)
        current_time += travel_segments[k][i]
        x.append(current_time)
        y.append(current_load / max_load)

    ax.plot(x, y, label = str(k) + ": " + str(max_load))

def plot_vehicle_loads(data, sol, travel_segments, load_use):

    fig, ax = plt.subplots(len(data["K"]), squeeze= False, sharex = True)
    for k in data["K"]:
        plot_vehicle_load(data, sol, travel_segments, load_use, k, ax[k][0])
    #ax.legend()
    plt.show()

def get_load_use(data, sol, vehicle_routes):

    load_use_table = {}
    for k, route in vehicle_routes.items():

        load_use = []
        if len(route) > 1:
            for s in range(1, len(route)):
                i = route[s - 1]
                j = route[s]
                current_load = 0
                for p in data["P"]:
                    if (p, i, j, k) in sol["q"]:
                        current_load += sol["q"][p, i, j, k] * data["omega"][p]
                load_use.append(current_load)
        load_use_table[k] = load_use

    return load_use_table

def get_total_product(data, sol):
    '''
    Get a dictionary
    table[location][product] = M[p,i,:]
    '''
    locations = []
    types = ["F", "D", "S", "C"]
    for type in types:
        locations += data[type]
    products = data["P"]
    table = {locations[i]: {products[j]: 0 for j in products} for i in locations}

    for p,i,k in sol["m"].keys():
        table[i][p] += sol["m"][p,i,k]

    return table

def plot_travel_distance_pie(data, sol, travel_segments):

    travel_distance = get_travel_distance(data, sol, travel_segments)
    travel_distance_array = [distance for distance in travel_distance.values()]
    labels = [str(k) for k in travel_distance.keys()]
    plt.pie(travel_distance_array, labels=labels)
    plt.show()

def get_vehicle_routes(data, sol):
    '''
    data: data dictionary
    sol: solution dictionaty
    returns: vehicle_routes, vehicle_tours_x, vehicle_tours_y
    '''
    vehicle_tours_x = {k: [] for k in data["K"]}
    vehicle_tours_y = {k: [] for k in data["K"]}
    vehicle_order = {k: {} for k in data["K"]}
    vehicle_routes = {k: [] for k in data["K"]}

    for tup, w in sol["w"].items():
        i, j, k = tup
        vehicle_order[k][i] = j

    for loc in data["V_i"].keys():
        for k in data["V_i"][loc]:
            points_x = [data["XY"][loc, 0]]
            points_y = [data["XY"][loc, 1]]
            locations = [loc]
            curr = loc
            while curr in vehicle_order[k]:
                new = vehicle_order[k][curr]
                del vehicle_order[k][curr]
                curr = new
                points_x.append(data["XY"][curr, 0])
                points_y.append(data["XY"][curr, 1])
                locations.append(curr)

            vehicle_tours_x[k] = points_x
            vehicle_tours_y[k] = points_y
            vehicle_routes[k] = locations

    return vehicle_routes, vehicle_tours_x, vehicle_tours_y

if __name__ == "__main__":
    """EXECUTION"""
    soldir = os.path.join(os.path.curdir, 'solutions')
    datadir = os.path.join(os.path.curdir, 'data')
    # Execute for 1 instance
    # file = 'v8-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    file = 'solution milp v1-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    datafile = "v1-city-n15-f2-d1-s4-c8-p1-v1.xlsx"
    sol = ReadSolution(soldir, file)
    data = ReadData(datadir, datafile)
    plot_solution(data, sol)

    plt.show()

    vehicle_routes, vehicle_tours_x, vehicle_tours_y = get_vehicle_routes(data, sol)
    travel_segments = get_travel_segments(data, sol, vehicle_routes)
    load_use = get_load_use(data, sol, vehicle_routes)
    average_load = get_average_load(data, sol, travel_segments, load_use)
    print(travel_segments)
    print(load_use)
    print(average_load)
    plot_vehicle_loads(data, sol, travel_segments, load_use)
    plot_travel_distance_pie(data, sol, vehicle_routes)
    help(plt.subplots)