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

    #Read data
    variables = ["q", "w", "u", "y", "m"]
    indices = ["p", "i", "j", "k"]
    print(os.path.join(datadir, file))
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

    "First plot the data"
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

    #Now create the routes
    #p is color
    #i,j is position
    #v is vehicle type
    V = ["b", "g", "r", "c", "m", "k"]
    P = ["--", ":", "-.", "-"]
    print(sol["q"])
    vehicle_order, vehicle_tours_x, vehicle_tours_y = get_vehicle_routes(data, sol)
    print(vehicle_order)
    print(vehicle_tours_x)
    print(vehicle_tours_y)
    for k in vehicle_tours_x.keys():
        ax.plot(vehicle_tours_x[k], vehicle_tours_y[k], V[k], label=str(k))

    ax.legend()
    plt.show()
    "Then plot the solution"
    print(get_total_product(data, sol))
    print(get_total_demand(data, sol))
    return 0

def get_total_demand(data, sol):

    table = {}
    for location in data["DEM"]:
        table[location] = {j: data["DEM"][location][j] for j in range(len(data["P"]))}

    return table


def get_total_product(data, sol):

    locations = []
    types = ["F", "D", "S", "C"]
    for type in types:
        locations += data[type]
    products = data["P"]
    table = {locations[i]: {products[j]: 0 for j in products} for i in locations}

    for p,i,k in sol["m"].keys():
        table[i][p] += sol["m"][p,i,k]

    return table

def get_vehicle_routes(data, sol):

    vehicle_tours_x = {k: [] for k in data["K"]}
    vehicle_tours_y = {k: [] for k in data["K"]}
    vehicle_order = {k: {} for k in data["K"]}
    vehicle_routes = {k: [] for k in data["K"]}

    for tup, w in sol["w"].items():
        i, j, k = tup
        vehicle_order[k][i] = j
    print(vehicle_order)

    for loc in data["V_i"].keys():
        for k in data["V_i"][loc]:
            print(k, loc)
            points_x = [data["XY"][loc, 0]]
            points_y = [data["XY"][loc, 1]]
            locations = [loc]
            curr = loc
            while curr in vehicle_order[k]:
                print("Locations", curr)
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

