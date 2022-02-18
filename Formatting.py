'''
Make the formatting changes to turn the foreign instances into our format
'''

import pandas as pd
import os
import numpy as np
import scipy as sc
from scipy.spatial import distance_matrix

def UchoaToData(path, vrp_file, sol_file):

    with open(os.path.join(path, vrp_file), "r") as f:
        text = f.readlines()
        info = {}
        points_x = {}
        points_y = {}
        demand = {}
        depots = []
        reading_points = False
        reading_demand = True
        reading_depot = False
        for line in text:
            if ":" in line:
                a, b = line.split(":")
                info[a[:-1]] = b
            elif "NODE_COORD_SECTION" in line:
                reading_points = True
            elif reading_points and "DEMAND_SECTION" not in line:
                i, x, y = line.split("\t")
                points_x[int(i)] = int(x)
                points_y[int(i)] = int(y)
            elif "DEMAND_SECTION" in line:
                reading_demand = True
                reading_points = False
            elif reading_demand and "DEPOT_SECTION" not in line:
                i, d = line.split("\t")
                demand[int(i)] = int(d)
            elif "DEPOT_SECTION" in line:
                reading_demand = False
                reading_points = False
                reading_depot = True
            elif reading_depot and not "-1" in line:
                depots.append(int(line))
            elif "-1" in line:
                reading_depot = False
                break

        with open(os.path.join(path, sol_file), "r") as f:
            #open solution and read the amount of vehicles

        info["Capacity"] = int(info["CAPACITY"])
        info["Dimension"] = int(info["DIMENSION"])
        print(info)


        #Translate to data
        data = {}
        data["D"] = depots
        data["XY"] = np.array([[points_x[i], points_y[i]] for i in points_x.keys()])

        r = distance_matrix(data["XY"], data["XY"])
        data["r"] = r
        print(data)

if __name__ == "__main__":
    '''
    First part is to run all the instances and save them as solution files
    '''
    import os
    from scripts.milp import ExecuteMultiEchelonFromData, ReadData
    import time
    import Plotting
    directory = "data"
    datadir = os.path.join(os.path.curdir, 'data')
    soldir = os.path.join(os.path.curdir, 'solutions')
    plotdir = None
    uchoadir = os.path.join(os.path.curdir, 'uchoa')


    for filename in os.listdir(uchoadir):

        if filename.endswith(".vrp"):

            UchoaToData(uchoadir, filename, filename.replace(".vrp", ".sol"))


            break
