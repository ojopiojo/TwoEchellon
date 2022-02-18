'''
Make the formatting changes to turn the foreign instances into our format
'''

import pandas as pd
import os
import numpy as np
import scipy as sc
from scipy.spatial import distance_matrix
from Plotting import *
from scripts.milp import *

def UchoaToDataNoSOl(path, vrp_file):


    data = {}
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
                i, d = line.split("\t", maxsplit=1)
                demand[int(i)] = [int(d)]
            elif "DEPOT_SECTION" in line:
                reading_demand = False
                reading_points = False
                reading_depot = True
            elif reading_depot and not "-1" in line:
                depots.append(int(line))
            elif "-1" in line:
                reading_depot = False
                break



        info["Capacity"] = int(info["CAPACITY"])
        info["Dimension"] = int(info["DIMENSION"])
        print(info)

        points_x[0] = points_x[depots[0]]
        points_y[0] = points_y[depots[0]]
        n_nodes = len(demand.keys()) + 1
        n_vehicles = n_nodes // 4

        data["XY"] = np.array([[points_x[i], points_y[i]] for i in points_x.keys()])
        data["F"] = [0]
        data["D"] = depots
        data["S"] = []
        data["C"] = [i for i in demand.keys() if i not in depots + [0]]
        data["P"] = [0]
        data["P_f"] = {0: [0]}
        #We leave 10% margin on number of vehicles
        data["K"] = [i for i in range(n_vehicles)]
        data["V_i"] = {depots[0] : data["K"]}
        data["DEM"] = demand
        data["Lambd"] = {i: 10000000 for i in depots}
        data["Omega"] = {i: 10000000 for i in depots}
        data["Phi"] = {i : info["Capacity"] for i in range(n_vehicles)}
        data["Theta"] = {i : 1000 for i in range(n_vehicles)}
        data["nu"] = [1]
        data["omega"] = [1]
        data["omegaeff"] = [1]
        data["rho"] = {i : 1 for i in range(n_vehicles)}
        data["delta"] = {i : 1000 for i in range(n_vehicles)}
        data["gamma"] = {i : 0 for i in range(n_vehicles)}
        data["epsil"] = {}
        data["r"] = distance_matrix(data["XY"], data["XY"])
        data["LEZ"] = [0 for i in range(n_nodes)]
        data["vehictype"] = {i : "truck" for i in range(n_vehicles)}
        data["city"] = {i : "BR" for i in range(n_nodes)}
        data["A"] = np.array([[1 for i in range(n_vehicles)] for j in range(n_nodes)])

        return data


def UchoaToData(path, vrp_file, sol_file):


    data = {}
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
                i, d = line.split("\t", maxsplit=1)
                demand[int(i)] = [int(d)]
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
            n_vehicles = int((len(f.readlines()) - 1) * 1.1)

        info["Capacity"] = int(info["CAPACITY"])
        info["Dimension"] = int(info["DIMENSION"])
        print(info)

        points_x[0] = points_x[depots[0]]
        points_y[0] = points_y[depots[0]]
        n_nodes = len(demand.keys()) + 1

        data["XY"] = np.array([[points_x[i], points_y[i]] for i in points_x.keys()])
        data["F"] = [0]
        data["D"] = depots
        data["S"] = []
        data["C"] = [i for i in demand.keys() if i not in depots + [0]]
        data["P"] = [0]
        data["P_f"] = {0: [0]}
        #We leave 10% margin on number of vehicles
        data["K"] = [i for i in range(n_vehicles)]
        data["V_i"] = {depots[0] : data["K"]}
        data["DEM"] = demand
        data["Lambd"] = {i: 10000000 for i in depots}
        data["Omega"] = {i: 10000000 for i in depots}
        data["Phi"] = {i : info["Capacity"] for i in range(n_vehicles)}
        data["Theta"] = {i : 1000 for i in range(n_vehicles)}
        data["nu"] = [1]
        data["omega"] = [1]
        data["omegaeff"] = [1]
        data["rho"] = {i : 1 for i in range(n_vehicles)}
        data["delta"] = {i : 1000 for i in range(n_vehicles)}
        data["gamma"] = {i : 0 for i in range(n_vehicles)}
        data["epsil"] = {}
        data["r"] = distance_matrix(data["XY"], data["XY"])
        data["LEZ"] = [0 for i in range(n_nodes)]
        data["vehictype"] = {i : "truck" for i in range(n_vehicles)}
        data["city"] = {i : "BR" for i in range(n_nodes)}
        data["A"] = np.array([[1 for i in range(n_vehicles)] for j in range(n_nodes)])

        return data

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
    soldir = os.path.join(os.path.curdir, 'uchoa_solutions')
    plotdir = None
    uchoadir = os.path.join(os.path.curdir, 'ArtificialPoints')


    for filename in os.listdir(uchoadir):


        if filename.endswith(".vrp"):
            data = UchoaToDataNoSOl(uchoadir, filename)
            ti = perf_counter()
            q_final, w_final, u_final, y_final, m_final, Opt = ExecuteMultiEchelon(data)
            dt = ti - perf_counter()
            save_solution(uchoadir, "milp solution " + filename.replace(".vrp", ".xlsx"), dt, q_final, w_final, u_final, y_final, m_final, Opt)


