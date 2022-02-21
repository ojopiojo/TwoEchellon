#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:39:33 2020

@author: cristian
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from gurobipy import *
from matplotlib import cm
from time import time
from scripts.utils import save_solution
#from shapely.geometry import Point, shape
#from numpy.random import uniform
#from collections import Counter

"""
DistanceBetweenNodes: Given a set of coordinates XY, computes the euclidean distance between all nodes
"""

def DistanceBetweenNodes(XY):
    n = XY.shape[0]
    # Distance between points
    r = np.zeros((n, n))
    for i in range(n):
        for j in range(i,n):
            form = np.around(np.sqrt((XY[i,0] - XY[j,0])**2 + (XY[i,1] - XY[j,1])**2))
            r[i,j] = form
            r[j,i] = form
    return r

"""
ReadData: function for getting data from a .xlsx file
Returns a dictionary with all the data extracted from the .xlsx file
"""
def ReadData(datadir, file):
    # Nodes
    df = pd.read_excel(os.path.join(datadir, file), sheet_name = 'Nodes')
    XY = df[['X','Y']].values
    F = df[df['Type'] == 'F'].index.tolist()
    D = df[df['Type'] == 'D'].index.tolist()
    S = df[df['Type'] == 'S'].index.tolist()
    C = df[df['Type'] == 'C'].index.tolist()
    LEZ = dict(zip(df.index.tolist(),df['LEZ'].tolist()))
    city = dict(zip(df.index.tolist(),df['City'].tolist()))
    # Products
    df = pd.read_excel(os.path.join(datadir, file), sheet_name = 'Products')
    P = df['Product'].tolist()
    nu = df['Volume'].tolist()
    omega = df['Weight'].tolist()
    omegaeff = df['Weight eff'].tolist()
    P_f = {}
    for f in F:
        P_f[f] = df[df['Firm'] == f]['Product'].tolist()
    # Demands
    df = pd.read_excel(os.path.join(datadir, file), sheet_name = 'Demands') 
    DEM = {}
    for c in C:
        DEM[c] = df[df['Customer'] == c]['Demand'].tolist()
    # Depots cap.
    df = pd.read_excel(os.path.join(datadir, file), sheet_name = 'Depots cap.')
    Lambd = {}
    Omega = {}
    epsil = {}
    for i in range(df.shape[0]):
        d = int(df['Depot'].iloc[i])
        Lambd[d] = df['Lambd'].iloc[i]
        Omega[d] = df['Omega'].iloc[i]
        epsil[d] = df['epsil'].iloc[i]
    # Vehicles
    df = pd.read_excel(os.path.join(datadir, file), sheet_name = 'Vehicles')
    K = df['Vehicle'].tolist()
    V_i = {}
    Phi = {}
    Theta = {}
    rho = {}
    delta = {}
    gamma = {}
    vehictype = {}
    DcupS = D+S
    for d in DcupS:
        V_i[d] = df[df['Depot'] == d]['Vehicle'].tolist()
        for k in V_i[d]:
            Phi[k] = df[df['Vehicle'] == k]['Phi'].sum()
            Theta[k] = df[df['Vehicle'] == k]['Theta'].sum()
            rho[k] = df[df['Vehicle'] == k]['rho'].sum()
            delta[k] = df[df['Vehicle'] == k]['delta'].sum()
            gamma[k] = df[df['Vehicle'] == k]['gamma'].sum()
            vehictype[k] = df[df['Vehicle'] == k]['VehicleType'].iloc[0]
    r = DistanceBetweenNodes(XY)
    """
    DATA DICTIONARY
    """
    data = {}
    data['XY'] = XY
    data['F'] = F
    data['D'] = D
    data['S'] = S
    data['C'] = C
    data['P'] = P
    data['P_f'] = P_f
    data['K'] = K
    data['V_i'] = V_i
    data['DEM'] = DEM
    data['Lambd'] = Lambd
    data['Omega'] = Omega
    data['Phi'] = Phi
    data['Theta'] = Theta
    data['nu'] = nu
    data['omega'] = omega
    data['omegaeff'] = omegaeff
    data['rho'] = rho
    data['delta'] = delta
    data['gamma'] = gamma
    data['epsil'] = epsil
    data['r'] = r
    data['LEZ'] = LEZ
    data['vehictype'] = vehictype
    data['city'] = city
    A = np.ones((len(F+D+S+C), len(K)), dtype=int)
    for s in S:
        for k in V_i[s]:
            for s1 in S:
                if s1 != s:
                    # Bikes aren't shared between satellites
                    A[s1,k] = 0
            for n in F+D+C:
                # Bikes only visit nodes from the same city
                if vehictype[k] == 'bike' and city[s] != city[n]:
                    A[n,k] = 0
                # Non eco vehicles aren't allowed in LEZ points
                if vehictype[k] != 'bike' and LEZ[n] > 0:
                    A[n,k] = 0
    for d in D:
        for k in V_i[d]:
            for d1 in D:
                if d1 != d:
                    # Vehicles aren't shared between delivering
                    A[d1,k] = 0
            for n in F+S+C:
                # Non eco vehicles aren't allowed in LEZ points
                if vehictype[k] != 'bike' and LEZ[n] > 0:
                    A[n,k] = 0
    data['A'] = A
    return data

"""
MATH HEURISTIC FUNCTIONS
Here are all the steps for solving the multi-echelon multi-vehicle problem
"""
def GreedyRoutingForServingCost(d0, W0, NodesToVisit, WeightNodes, gamma_k, rho_k, r):
    # This function estimates the serving cost via greedy routing
    VisitedNodes = [d0]
    PendingNodes = [n for n in NodesToVisit]
    TotalCost = 0
    CumulatedWeight = W0
    # Initial case: select the first node to visit
    i = d0
    ArcCost = np.inf
    for j in PendingNodes:
        CurrentCost = r[i,j]*(gamma_k*CumulatedWeight + rho_k)
        if CurrentCost < ArcCost:
            ArcCost = CurrentCost
            j_ = j
    TotalCost = TotalCost + ArcCost
    VisitedNodes.append(j_)
    CumulatedWeight = CumulatedWeight + WeightNodes[j_]
    PendingNodes = [n for n in PendingNodes if n not in VisitedNodes]
    i = j_
    # rest of the cases
    while PendingNodes:
        i = j_
        ArcCost = np.inf
        for j in PendingNodes:
            CurrentCost = r[i,j]*(gamma_k*CumulatedWeight + rho_k)
            if CurrentCost < ArcCost:
                ArcCost = CurrentCost
                j_ = j
        TotalCost = TotalCost + ArcCost
        VisitedNodes.append(j_)
        CumulatedWeight = CumulatedWeight + WeightNodes[j_]
        PendingNodes = [n for n in PendingNodes if n not in VisitedNodes]
    # return a tuple with the last node visited and the total cost
    return j_, TotalCost

def GetMinimalLoadCostF1(r, i, gamma, Weight_cf, rho, FminFw, D, V_i):
    loadcostf1 = 0
#     for f in FminFw:
#         gamma_kf = max([gamma[v] for v in V_i[f]])
#         rho_kf = max([rho[v] for v in V_i[f]])
#         cost_f = r[f,i]*(gamma_kf*Weight_cf[f] + rho_kf)
#         cost_d = np.inf
#         for d in D:
#             gamma_kd = max([gamma[v] for v in V_i[d]])
#             rho_kd = max([rho[v] for v in V_i[d]])
#             cost_d_ = r[f,d]*(gamma_kf*Weight_cf[f] + rho_kf) + r[d,i]*(gamma_kd*Weight_cf[f] + rho_kd)
#             if cost_d_ < cost_d:
#                 cost_d = cost_d_
#         loadcostf1 = loadcostf1 + min(cost_f, cost_d)
    return loadcostf1

def GetBestDeliveringCost(r, i, gamma, Weight_cf, rho, FirmsToVisit, D, V_i):
    cost_d = np.inf
    for d in D:
        gamma_kd = max([gamma[v] for v in V_i[d]])
        rho_kd = max([rho[v] for v in V_i[d]])
        f0, cost_d_ = GreedyRoutingForServingCost(d, 0, FirmsToVisit, Weight_cf, gamma_kd, rho_kd, r)
        cost_d_ = cost_d_ + r[f0,i]*(sum([gamma_kd*Weight_cf[f] for f in FirmsToVisit]) + rho_kd)
        if cost_d_ < cost_d:
            cost_d = cost_d_
    return cost_d 

def Inter(list1, list2):
    return [i for i in list1 if i in list2]

def GetFeasibleCombinationsForVehicles(minlen, nodes, Theta, Phi, WeightClient, VolClient, banned):
    result = []
    for i in range(len(nodes), minlen, -1):
        for seq in itertools.combinations(nodes, i):
            if sum([WeightClient[c] for c in seq]) <= Theta and sum([VolClient[c] for c in seq]) <= Phi:
                result.append(list(seq))
    return [r for r in result if not Inter(r,banned)]

def GetBestListOfNodes(result, VolClient, WeightClient, banned):
    prod = 0
    bestlist = []
    for l in result:
        if l not in banned:
            vol_l = sum([VolClient[c] for c in l])
            weight_l = sum([WeightClient[c] for c in l])
            if vol_l*weight_l > prod:
                prod = vol_l*weight_l
                bestlist = l
    return bestlist

def GetRoutingList(k,d0,N,w_final):
    routing = []
    test = int(sum([w_final[i,j,k] for i in N for j in N]))
    if test > 2:
        routing_list = [d0]
        j = int(sum([w_final[d0,l,k]*l for l in N]))
        routing.append((d0,j))
        routing_list.append(j)
        while j != d0:
            i = j
            j = int(sum([w_final[i,l,k]*l for l in N]))
            routing_list.append(j)
            routing.append((i,j))
    elif test  == 2:
        j = int(sum([w_final[d0,l,k]*l for l in N]))
        routing = [(d0,j), (j,d0)]
        routing_list = [d0, j]
    else:
        routing = []
        routing_list = []
        ##print('empty route')
    return routing, routing_list

def CreateSetE(DEM, F, C, P_f):
    DictFirmCl = {}
    for c in C:
        listdem = []
        for f in F:
            listdem.append(min(sum([DEM[c][p] for p in P_f[f]]),1))
        DictFirmCl[c] = listdem
    listdem = []
    for key in DictFirmCl.keys():
        if DictFirmCl[key] not in listdem:
            listdem.append(DictFirmCl[key])
    DemVecCl = {}
    for l in range(len(listdem)):
        dem = listdem[l]
        DemVecCl[l] = [c for c in DictFirmCl.keys() if DictFirmCl[c] == dem]
    E_c = {}
    for key in DemVecCl.keys():
        l = DemVecCl[key]
        if len(l) % 2 != 0:
            l = l[:-1]
        for c in l:
            E_c[c] = [e for e in l if e != c]
    return E_c

def ConstructorForRouting(dictclass, d0, k, m_opt, x_opt, data):
    # Unpack data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']
    DEM = data['DEM']
    Lambd = data['Lambd']
    Omega = data['Omega']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    omegaeff = data['omegaeff']
    rho = data['rho']
    delta = data['delta']
    gamma = data['gamma']
    r = data['r']
    N = F+D+S+C
    N_ = []
    DEM_ = {}
    Q0 = []
    if dictclass[d0] != 'D':
        for p in P:
            valid_keys = [(p,j,k) for j in N if (p,j,k) in m_opt.keys()]
            q0_ = int(sum([m_opt[t] for t in valid_keys]))
            Q0.append(q0_)
    else:
        for p in P:
            valid_keys = [(p,j,k) for j in N if (p,j,k) in m_opt.keys()]
            valid_keys_f = [(p,f,k) for f in F if (p,f,k) in x_opt.keys()]
            q0_ = int(sum([m_opt[t] for t in valid_keys]) - sum([x_opt[t] for t in valid_keys_f]))
            Q0.append(q0_)
#    if dictclass[d0] == 'F':     
#        Q0 = [int(sum([m_opt[p,j,k] for j in N])) for p in P]
#    else:
#        Q0 = [int(sum([(m_opt[p,j,k])for j in N]) - sum([(x_opt[p,f,k])for f in F])) for p in P]
    Q0 = [max(q,0) for q in Q0]
    N_.append(d0)
    Nmin_i = [n for n in N if n != d0]
    DEM_[d0] = [int(m_opt.get((p,d0,k), 0)) for p in P]
    for j in Nmin_i:
        if dictclass[j] != 'F':
            tot = sum([m_opt.get((p,j,k), 0) for p in P])
        else:
            tot = sum([x_opt.get((p,j,k),0) for p in P])
        if tot > 0:
            N_.append(j)
            if dictclass[j] != 'F':
                DEM_[j] = [int(m_opt.get((p,j,k),0)) for p in P]
            else:
                DEM_[j] = [-int(x_opt.get((p,j,k),0)) for p in P]
    F_ = [f for f in F if f in N_]
    D_ = [d for d in D if d in N_]
    S_ = [s for s in S if s in N_]
    C_ = [c for c in C if c in N_]
    data_routing = {'Q0' : Q0,
                    'DEM': DEM_,
                    'N': N_,
                    'F' : F_,
                    'D': D_,
                    'S': S_,
                    'C': C_,
                    'P' : P,
                    'Phi' : Phi[k],
                    'Theta': Theta[k],
                    'nu' : nu,
                    'omega': omega,
                    'omegaeff': omegaeff,
                    'd0' : d0,
                    'gamma': gamma[k],
                    'rho' : rho[k],
                    'r' : r}
    return data_routing


def FlowRouting(data_routing):
    # Unpacking data
    F = data_routing['F']
    D = data_routing['D']
    S = data_routing['S']
    C = data_routing['C']
    P = data_routing['P']
    Phi = data_routing['Phi']
    Theta = data_routing['Theta']
    nu = data_routing['nu']
    omega = data_routing['omega']
    omegaeff = data_routing['omegaeff']
    gamma = data_routing['gamma']
    DEM = data_routing['DEM']
    d0 = data_routing['d0']
    Q0 = data_routing['Q0']
    rho = data_routing['rho']
    r = data_routing['r']
    # Auxiliary sets
    N = F+D+S+C
    NminC = F+D+S
    NminF = D+S+C
    Nmind0 = [n for n in N if n != d0]
    Nmind0_i = {}
    ScupCmind0 = [i for i in S+C if i != d0]
    for i in Nmind0:
        Nmind0_i[i] = [j for j in Nmind0 if j != i]
    # Consolidation of weight and volume
    Weight = {}
    Volume = {}
    WeightE = {}
    for i in N:
        try:
            Weight[i] = sum([DEM[i][p]*omega[p] for p in P])
        except:
            Weight[i] = 0
        try:
            WeightE[i] = sum([DEM[i][p]*omegaeff[p] for p in P])
        except:
            WeightE[i] = 0  
        try:
            Volume[i] = sum([DEM[i][p]*nu[p] for p in P])
        except:
            Volume[i] = 0
    #print(Q0)
    W0 = sum([Q0[p]*omega[p] for p in P])
    W0e = sum([Q0[p]*omegaeff[p] for p in P])
    ##print('W0 = ', W0)
    V0 = sum([Q0[p]*nu[p] for p in P])
    #print('V0 = ', V0, " Vol cap = ", Phi)
    #print('W0 = ', W0, " Weight cap = ", Theta)
    #print('W0 effective = ', W0e)
#    #print('N = ', N)
#    #print('Nmind0 = ', Nmind0)
    # Model start
    model = Model()
    model.setParam('OutputFlag', 0)
    # Decision variables
    q = {}
    for i in N:
        for j in N:
            q[i,j] = model.addVar(vtype = GRB.CONTINUOUS, name = 'q[%s,%s]' % (i,j))
    qe = {}
    for i in N:
        for j in N:
            qe[i,j] = model.addVar(vtype = GRB.CONTINUOUS, name = 'q[%s,%s]' % (i,j))
    v = {}
    for i in N:
        for j in N:
            v[i,j] = model.addVar(vtype = GRB.CONTINUOUS, name = 'v[%s,%s]' % (i,j))
    w = {}
    for i in N:
        for j in N:
            w[i,j] = model.addVar(vtype = GRB.BINARY, name = 'w[%s,%s]' % (i,j))
    e = {}
    for i in Nmind0:
        e[i] = model.addVar(vtype = GRB.INTEGER, name = 'e[%s]' % i)
    # Aux
    fc = model.addVar(vtype = GRB.CONTINUOUS, name = 'fc')
    ac = model.addVar(vtype = GRB.CONTINUOUS, name = 'ac')
    # Constraints
    # Flow
    model.addConstrs(
        quicksum(q[i,j] for i in N) - quicksum(q[j,l] for l in N) == Weight[j] for j in Nmind0
    )
    model.addConstrs(
        quicksum(qe[i,j] for i in N) - quicksum(qe[j,l] for l in N) == WeightE[j] for j in Nmind0
    )
    model.addConstrs(
        quicksum(v[i,j] for i in N) - quicksum(v[j,l] for l in N) == Volume[j] for j in Nmind0
    )
    model.addConstrs(
        quicksum(w[i,j] for i in N) == quicksum(w[j,l] for l in N) for j in N
    )
    model.addConstrs(
        q[i,j] <= Theta*w[i,j] for i in N for j in N
    )
    model.addConstrs(
        qe[i,j] <= 2*(Theta + Phi)*w[i,j] for i in N for j in N
    )
    model.addConstrs(
        v[i,j] <= Phi*w[i,j] for i in N for j in N
    )
    # Out
    model.addConstr(
        quicksum(q[d0,j] for j in N) == W0
    )
    model.addConstr(
        quicksum(qe[d0,j] for j in N) == W0e
    )
    model.addConstr(
        quicksum(v[d0,j] for j in N) == V0
    )
    # Back to depot OK with this one
    model.addConstr(
        quicksum(q[i,d0] for i in N) == Weight[d0]
    )
    model.addConstr(
        quicksum(qe[i,d0] for i in N) == WeightE[d0]
    )
    model.addConstr(
        quicksum(v[i,d0] for i in N) == Volume[d0]
    )
#     Node visiting OK with this one
    model.addConstrs(
        quicksum(w[i,j] for i in N) == 1 for j in N
    )
    # Node leaving OK with this one
    model.addConstrs(
        quicksum(w[i,j] for j in N) == 1 for i in N
    )
    # TMZ
    model.addConstrs(
        e[i] - e[j] + len(N)*w[i,j] <= len(N) - 1 for i in Nmind0 for j in Nmind0
    )
    model.addConstrs(
        e[i] >= 0 for i in Nmind0
    )
    Fmind0 = [f for f in F if f != d0]
    model.addConstrs(
        e[i] >= e[f] for i in ScupCmind0 for f in Fmind0
    )
    # Logic
    model.addConstrs(
        q[i,i] == 0 for i in N
    )
    # Logic
    model.addConstrs(
        qe[i,i] == 0 for i in N
    )
    model.addConstrs(
        v[i,i] == 0 for i in N
    )
    model.addConstrs(
        w[i,i] == 0 for i in N
    )
    # Capacity and arc utilization
    model.addConstr(
        fc == quicksum(quicksum(qe[i,j]*gamma*r[i,j] for i in N) for j in N)
    )
    model.addConstr(
        ac == quicksum(quicksum(w[i,j]*r[i,j]*rho for i in N) for j in N)
    )
    
    model.update()    
    model.__data = qe, q, w, v
    
    model.setObjective(fc + ac,
                       GRB.MINIMIZE)
    model.update()

    return model

"""STEP 4: Vehicle routing"""
def MultiEchelonRouting(data, x_opt, y_opt, m_opt, z_opt, u_opt):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']
    DEM = data['DEM']
    Lambd = data['Lambd']
    Omega = data['Omega']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    omegaeff = data['omegaeff']
    rho = data['rho']
    delta = data['delta']
    gamma = data['gamma']
    r = data['r']
    # Auxiliary sets
    N = F+D+S+C
    NminC = F+D+S
#    for p in P:
#        for c in C:
#            for k in K:
#                m_opt[p,c,k] = DEM[c][p]*z_opt[k,c]
    model = Model()
    dictclass = {}
    for f in F:
        dictclass[f] = 'F'
    for d in D:
        dictclass[d] = 'D'
    for s in S:
        dictclass[s] = 'S'
    for c in C:
        dictclass[c] = 'C'
    # DEFINITIVE DECISION VARIABLES
    q_final, qe_final, w_final, v_final = {}, {} ,{}, {}
    for i in N:
        for j in N:
            for k in K:
                q_final[i,j,k] = 0
                qe_final[i,j,k] = 0
                w_final[i,j,k] = 0
                v_final[i,j,k] = 0
    # Auxiliary dictionary for routes
    DictRoutes = {}
    DictRoutesList = {}
    # Auxiliary dictionary for subsets for each vehicle
    DictNodes = {}
    # Auxiliary dictionaries for remaining capacities
    Phi_, Theta_ = {}, {}
    Q0_, DEMS_ = {}, {}
    for k in K:
        DictRoutes[k] = []
        DictRoutesList[k] = []
        DictNodes[k] = {'F' : [], 'D' : [], 'S': [], 'C': []}
        Phi_[k] = Phi[k]
        Theta_[k] = Theta[k]
#    for d0 in NminC:
    #print('y_opt = ', y_opt)
    for d0 in D+S:
        ##print('Node: %s, Vehicles = %s' % (d0,V_i[d0]))
        for k in V_i[d0]:
            data_routing = ConstructorForRouting(dictclass, d0, k, m_opt, x_opt, data)
            if y_opt.get(k,0) > 0 and len(data_routing['N']) > 1:
                #print('data for routing vehicle %s' % k)
                #print(data_routing)
                model_rou = FlowRouting(data_routing)
                model_rou.optimize()
                qe, q, w, v = model_rou.__data
                q_rou = model_rou.getAttr('x', q)
                qe_rou = model_rou.getAttr('x', qe)
                w_rou = model_rou.getAttr('x', w)
                v_rou = model_rou.getAttr('x', v)
                F_ = data_routing['F']
                D_ = data_routing['D']
                S_ = data_routing['S']
                C_ = data_routing['C']
                N_ = F_ + D_ + S_ + C_
                Q0 = data_routing['Q0']
                DEM_ = data_routing['DEM']
                try:
#                     model_rou.##printAttr('X')
                    qe, q, w, v = model_rou.__data
                    q_rou = model_rou.getAttr('x', q)
                    qe_rou = model_rou.getAttr('x', qe)
                    w_rou = model_rou.getAttr('x', w)
                    v_rou = model_rou.getAttr('x', v)
                    for i in N_:
                        for j in N_:
                            try:
                                q_final[i,j,k] = q_rou[i,j]
                                qe_final[i,j,k] = qe_rou[i,j]
                                w_final[i,j,k] = w_rou[i,j]
                                v_final[i,j,k] = v_rou[i,j]
                            except:
                                q_final[i,j,k] = 0
                                w_final[i,j,k] = 0
                                v_final[i,j,k] = 0
                    Theta_[k] = Theta[k] - max([q_rou[i,j] for i in N_ for j in N_])
                    Phi_[k] = Phi[k] - max([v_rou[i,j] for i in N_ for j in N_])
                    DictRoutes[k], DictRoutesList[k] = GetRoutingList(k,d0,N,w_final)
                    DictNodes[k] = {'F' : F_, 'D' : D_, 'S': S_, 'C': C_}
                    Q0_[k] = Q0
                    DEMS_[k] = DEM_
                except:
                    pass
                    ##print('ERROR IN VEHICLE %s' % k)
            else:
                for i in N:
                    for j in N:
                        q_final[i,j,k] = 0
                        qe_final[i,j,k] = 0
                        v_final[i,j,k] = 0
                        w_final[i,j,k] = 0
    solution = {'q_final' : q_final,
                'qe_final' : qe_final,
                'v_final' : v_final,
                'w_final' : w_final, 
                'DictRoutes' : DictRoutes,
                'DictRoutesList' : DictRoutesList,
                'DictNodes' : DictNodes,
                'Theta_' : Theta_,
                'Phi_' : Phi_,
                'Q0_' : Q0_,
                'DEMS_' : DEMS_}  
    return solution

"""
AUXILIARY FUNCTIONS FOR HEURISTIC LOOP
"""

# Function that computes the freight cost for the route of a certain vehicle
def ComputeRouteCost(q, routing, k, gamma, r):
    return sum([q[i,j,k]*r[i,j]*gamma[k] for (i,j) in routing])

def DictFromListofTuples(listtuples):
    dic = {}
    for i,j in listtuples:
        dic[i] = j
    return dic

# Function that computes every routing cost for every vehicle that visits clients and satellites
def GetMaxRoutingCosts(N, K, depots, DictNodes, r, gamma, w_final, q_final):
    RC = {}
    Kfil = [k for k in K if DictNodes[k]['S'] and DictNodes[k]['C']] # Vehicles that visit satellites and clients
    for k in Kfil:
        routing, routing_list = GetRoutingList(k,depots[k],N,w_final)
        freightcost = ComputeRouteCost(q_final, routing, k, gamma, r)
        RC[k] = freightcost
    try:
        RC = DictFromListofTuples(sorted(RC.items(), key=lambda x: x[1], reverse=True))
    except:
        pass
    return RC



# Function that determines the "most expensive" destination in a route
def GetNode2Exclude(routing, q, qe, v, C, k, gamma, r, banlist):
    DictRouFreight = dict(zip(routing,[qe[i,j,k]*r[i,j]*gamma[k] for (i,j) in routing]))
    MaxCost = 0
    ex = None
    q_ex = None
    v_ex = None
    for t in routing:
        if t[1] in C and t[1] not in banlist:
            if DictRouFreight[t] > MaxCost:
                ex = t[1]
                MaxCost = DictRouFreight[t]
    # get freight from that node
    if ex != None:
        ant_ex = [t for t in routing if t[1] == ex][0][0]
        post_ex = [t for t in routing if t[0] == ex][0][1]
        q_ex = q[ant_ex,ex,k] - q[ex,post_ex,k]
        qe_ex = qe[ant_ex,ex,k] - qe[ex,post_ex,k]
        v_ex = v[ant_ex,ex,k] - v[ex,post_ex,k]
    return ex, q_ex, qe_ex, v_ex

def GetNode2ExcludeFromVs(routing, q, v, C, k, gamma, r, banlist):
    DictRouFreight = dict(zip(routing,[q[i,j,k]*r[i,j]*gamma[k] for (i,j) in routing]))
    MaxCost = 0
    ex = None
    q_ex = None
    v_ex = None
    if len(routing) > 2:
        for t in routing:
            if t[1] in C and t[1] not in banlist:
                if DictRouFreight[t] > MaxCost:
                    ex = t[1]
                    MaxCost = DictRouFreight[t]
    elif len(routing) == 2:
        ##print('Route of lenght 2')
        ex = routing[0][1]
    else:
        pass
    if ex != None:
        ant_ex = [t for t in routing if t[1] == ex][0][0]
        post_ex = [t for t in routing if t[0] == ex][0][1]
        q_ex = q[ant_ex,ex,k] - q[ex,post_ex,k]
        v_ex = v[ant_ex,ex,k] - v[ex,post_ex,k]
    return ex, q_ex, v_ex

# Function that takes a route and adds a node to that route
def ReroutingAdd(routing, tremove, tadd):
    rerouting = []
    for t in routing:
        if t == tremove:
            rerouting = rerouting + tadd #tadd is a list of 2 tuples
        else:
            rerouting.append(t)
    return rerouting

# Function that takes a route and removes a node from that route
def ReroutingRemove(routing, ex):
    rerouting = []
    t_aux = [None,None]
    for t in routing:
        flag = True
        if t[1] == ex:
            t_aux[0] = t[0]
            flag = False
        if t[0] == ex:
            t_aux[1] = t[1]
            flag = False
        try:
            if sum(t_aux) > 0:
                rerouting.append(tuple(t_aux))
                t_aux = [None,None]
        except:
            pass
        if flag:
            rerouting.append(t)
    return rerouting

# Function that decides which is the best way for adding a node to a route
def MinRouteVariation(Vsat, ex, r, DictRoutes):
    MinDist = np.inf
    for k in Vsat:
        d0 = DictRoutes[k][0]
        for t in DictRoutes[k]:
            i1 = t[0]
            j1 = t[1]
            dist = r[i1,ex] + r[j1,t[1]]
            if i1 != d0 and j1 != d0:
                if dist < MinDist:
                    ks = k
                    i = t[0]
                    l = t[1]
                    tremove = (i,l)
                    tadd = [(i,ex), (ex,l)]
                    MinDist = dist
    # Rerouting
    rerouting = ReroutingAdd(DictRoutes[ks], tremove, tadd)
    return ks, rerouting

# Function that decides which satellite will receive the freight from the excluded node
def SelectSatellite(ex, q_ex, v_ex, Sk, V_i, cdv, cdw, Phi_, Theta_, r, DictRoutes):
    sat = None
    ks = None
    rerouting = None
    MinDist = np.inf
    for s in Sk:
        if q_ex <= cdw[s] and v_ex <= cdv[s]:
            Vsat = [k for k in V_i[s] if Theta_[k] >= q_ex and Phi_[k] >= v_ex]
            if len(Vsat) > 0:
                if r[s,ex] < MinDist:
                    sat = s
                    MinDist = r[s,ex]
                    ks, rerouting = MinRouteVariation(Vsat, ex, r, DictRoutes)
    return sat, ks, rerouting

# Function for recomputing the freight for routing
def RecomputeFreightEx(q_final,w_final, N, k, ex, q_ex, sat, routing, gamma, r):
    routing_list = [routing[0][0]]
    for t in routing:
        routing_list.append(t[1])
    flag_ex = 0
    q_rec = {}
    for i in routing_list[:-1]:
        j = int(sum([w_final[i,j,k]*j for j in N]))
        if j == ex:
            ex_ant = i
            flag_ex = q_ex
        elif j == sat:
            if i == ex:
                q_rec[ex_ant,j,k] = q_final[i,j,k] + flag_ex
            else:
                q_rec[i,j,k] = q_final[i,j,k] + flag_ex
            flag_ex = 0
        else:
            if i == ex:
                q_rec[ex_ant,j,k] = q_final[i,j,k] + flag_ex
            else:
                q_rec[i,j,k] = q_final[i,j,k] + flag_ex
    rerouting = ReroutingRemove(routing, ex)
    return ComputeRouteCost(q_rec, rerouting, k, gamma, r)

# Function for recomputing the freight for routing
def RecomputeFreightAdd(q_final, N, k, ex, q_ex, rerouting, gamma, r):
    flag_ex = q_ex
    q_rec = {}
    for t in rerouting:
        i = t[0]
        j = t[1]
        if j == ex:
            ex_ant = i
        else:
            if i == ex:
                q_rec[ex_ant,i,k] = q_final[ex_ant,j,k] + flag_ex
                flag_ex = 0
                q_rec[i,j,k] = q_final[ex_ant,j,k] + flag_ex
            else:
                q_rec[i,j,k] = q_final[i,j,k] + flag_ex
    return ComputeRouteCost(q_rec, rerouting, k, gamma, r)

def RecomputeFreightExFromKs(q_final,w_final, N, k, ex, q_ex, routing, gamma, r):
    ##print('Removing %s from freigh of vehicle %s' %(q_ex, k))
    # Function for recomputing freight
    if len(routing) > 2:
        routing_list = [routing[0][0]]
        for t in routing:
            routing_list.append(t[1])
        flag_ex = q_ex
        q_rec = {}
        for i in routing_list[:-1]:
            j = int(sum([w_final[i,j,k]*j for j in N]))
            if j == ex:
                ex_ant = i
                flag_ex = 0
            else:
                if i == ex:
                    q_rec[ex_ant,j,k] = q_final[i,j,k] - flag_ex
                else:
                    q_rec[i,j,k] = q_final[i,j,k] - flag_ex
        rerouting = ReroutingRemove(routing, ex)
        cost = ComputeRouteCost(q_rec, rerouting, k, gamma, r)
    else:
        cost = 0
    return cost

def RecomputeFreightAddToKd(q_final, N, k, ex, q_ex, sat, rerouting, gamma, r):
    # Function for recomputing the freight for routing
    ##print('Adding %s to freight of vehicle %s' %(q_ex, k))
    ##print('Rerouting: ', rerouting)
    q_or = {}
    for (i,j) in rerouting:
        try:
            q_or[i,j,k] = q_final[i,j,k]
        except:
            pass
    routing_list = [rerouting[0][0]]
    for t in rerouting:
        routing_list.append(t[1])
    if routing_list.index(ex) < routing_list.index(sat):
        flag_ex = 0
        q_rec = {}
        for t in rerouting:
            i = t[0]
            j = t[1]
            if j == ex:
                ex_ant = i
            else:
                if i == ex:
                    q_rec[ex_ant,ex,k] = q_final[ex_ant,j,k]
                    flag_ex = q_ex
                    q_rec[ex,j,k] = q_final[ex_ant,j,k] - flag_ex
                elif i == sat:
                    flag_ex = 0
                    q_rec[sat,j,k] = q_final[sat,j,k] - flag_ex
                else:
                    q_rec[i,j,k] = q_final[i,j,k] - flag_ex
    else:
        flag_ex = 0
        q_rec = {}
        for t in rerouting:
            i = t[0]
            j = t[1]
            if j == ex:
                ex_ant = i
            else:
                if i == ex:
                    q_rec[ex_ant,ex,k] = q_final[ex_ant,j,k]
                    q_rec[ex,j,k] = q_final[ex_ant,j,k] - flag_ex
                    flag_ex = 0
                elif i == sat:
                    flag_ex = q_ex
                    q_rec[sat,j,k] = q_final[sat,j,k] + flag_ex
                else:
                    q_rec[i,j,k] = q_final[i,j,k] + flag_ex
    ##print('Q nuevo: ', q_rec)
    return ComputeRouteCost(q_rec, rerouting, k, gamma, r)

def ImproveOptimalSwapKdKs(RCVd, data, cdv, cdw,  DictRoutes, DictRoutesList, DictNodes, DEMS_, Q0_, q_final, qe_final, v_final, w_final, Phi_, Theta_, depots):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    V_i = data['V_i']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    rho = data['rho']
    gamma = data['gamma']
    r = data['r']
    A = data['A']
    N = F+D+S+C
    banlist = []
    for kd in RCVd.keys():
        flag_feasible = True
        flag_descend = True
        while flag_feasible and flag_descend:
            # Get node to exclude REMARK: IS ALWAYS A CLIENT!!
            try:
                ex, q_ex, qe_ex, v_ex = GetNode2Exclude(DictRoutes[kd], q_final, qe_final, v_final, C, kd, gamma, r, banlist)
                # Get satellite
                sat, ks, rerouting_ks = SelectSatellite(ex, q_ex, v_ex, DictNodes[kd]['S'], V_i, cdv, cdw, Phi_, Theta_, r, DictRoutes)
            except:
                sat = None
            # If there is a satelite...
            if sat != None:
                IncumbentCost = np.inf
                # PrevCost: routing cost for kd and ks without changes
                PrevCost = ComputeRouteCost(qe_final, DictRoutes[kd], kd, gamma, r) + ComputeRouteCost(qe_final, DictRoutes[ks], ks, gamma, r)
                Costkd = RecomputeFreightEx(qe_final,w_final, N, kd, ex, qe_ex, sat, DictRoutes[kd], gamma, r)
                Costks = RecomputeFreightAdd(qe_final, N, ks, ex, qe_ex, rerouting_ks, gamma, r)
                IncumbentCost = Costkd + Costks
                if A[ex,ks] < 0:
                    IncumbentCost = np.inf
                ##print('Incumbent: ', IncumbentCost, ' previous: ', PrevCost)
                if IncumbentCost <= PrevCost:
                    # Modify nodes for kd and ks
                    ##print('Removing %s from the route of vehicle %s' % (ex,kd))
                    DictNodes[kd]['C'] = [c for c in DictNodes[kd]['C'] if c != ex]
                    DictNodes[ks]['C'] = DictNodes[ks]['C'] + [ex]
                    # Create entry for exchanged node
                    DEMS_[ks][ex] = [0 for p in P]
                    # Correct demand for excluded node
                    for p in P:
                        aux = DEMS_[kd][ex][p]
                        DEMS_[kd][sat][p] = DEMS_[kd][sat][p] + aux
                        Q0_[ks][p] = Q0_[ks][p] + aux
                        DEMS_[ks][ex][p] = aux
                        cdv[sat] = cdv[sat] + aux*nu[p]
                        cdw[sat] = cdw[sat] + aux*omega[p]
                    del DEMS_[kd][ex]

                    # Re routing for kd
                    ##print('RE ROUTING FOR VEHICLE %s' % kd)
                    F_ = DictNodes[kd]['F']
                    D_ = DictNodes[kd]['D']
                    S_ = DictNodes[kd]['S']
                    C_ = DictNodes[kd]['C']
                    N_ = F_ + D_ + S_ + C_
                    data_routing = {'Q0' : Q0_[kd],
                                    'DEM': DEMS_[kd],
                                    'N': N_,
                                    'F' : F_,
                                    'D': D_,
                                    'S': S_,
                                    'C': C_,
                                    'P' : P,
                                    'Phi' : Phi[kd],
                                    'Theta': Theta[kd],
                                    'nu' : nu,
                                    'omega': omega, 
                                    'd0' : depots[kd],
                                    'gamma': gamma[kd],
                                    'rho' : rho[kd],
                                    'r' : r}
                    model_rou = FlowRouting(data_routing)
                    model_rou.optimize()
                    try:
                        qe, q, w, v = model_rou.__data
                        q_rou = model_rou.getAttr('x', q)
                        qe_rou = model_rou.getAttr('x', qe)
                        w_rou = model_rou.getAttr('x', w)
                        v_rou = model_rou.getAttr('x', v)
                        for i in N_:
                            for j in N_:
                                try:
                                    q_final[i,j,kd] = q_rou[i,j]
                                    qe_final[i,j,kd] = qe_rou[i,j]
                                    w_final[i,j,kd] = w_rou[i,j]
                                    v_final[i,j,kd] = v_rou[i,j]
                                except:
                                    q_final[i,j,kd] = 0
                                    qe_final[i,j,kd] = 0
                                    w_final[i,j,kd] = 0
                                    v_final[i,j,kd] = 0
                        # Delete route for excluded node
                        for i in N_:
                            q_final[i,ex,kd] = 0
                            qe_final[i,ex,kd] = 0
                            w_final[i,ex,kd] = 0
                            v_final[i,ex,kd] = 0
                            q_final[ex,i,kd] = 0
                            qe_final[ex,i,kd] = 0
                            w_final[ex,i,kd] = 0
                            v_final[ex,i,kd] = 0  
                        Theta_[kd] = Theta[kd] - max([q_rou[i,j] for i in N_ for j in N_])
                        Phi_[kd] = Phi[kd] - max([v_rou[i,j] for i in N_ for j in N_])
                        DictRoutes[kd], DictRoutesList[kd] = GetRoutingList(kd,depots[kd],N,w_final)
                        DictNodes[kd] = {'F' : F_, 'D' : D_, 'S': S_, 'C': C_}
                        banlist.append(ex)
                    except:
                        pass
                        ##print('ERROR FOR VEHICLE %s' % kd)
                    ##print('RE ROUTING FOR VEHICLE %s' % ks)
                    F_ = DictNodes[ks]['F']
                    D_ = DictNodes[ks]['D']
                    S_ = DictNodes[ks]['S']
                    C_ = DictNodes[ks]['C']
                    N_ = F_ + D_ + S_ + C_
                    data_routing = {'Q0' : Q0_[ks],
                                    'DEM': DEMS_[ks],
                                    'N': N_,
                                    'F' : F_,
                                    'D': D_,
                                    'S': S_,
                                    'C': C_,
                                    'P' : P,
                                    'Phi' : Phi[ks],
                                    'Theta': Theta[ks],
                                    'nu' : nu,
                                    'omega': omega, 
                                    'd0' : depots[ks],
                                    'gamma': gamma[ks],
                                    'rho' : rho[ks],
                                    'r' : r}
                    model_rou = FlowRouting(data_routing)
                    model_rou.optimize()
                    try:
                        qe, q, w, v = model_rou.__data
                        q_rou = model_rou.getAttr('x', q)
                        qe_rou = model_rou.getAttr('x', qe)
                        w_rou = model_rou.getAttr('x', w)
                        v_rou = model_rou.getAttr('x', v)
                        for i in N_:
                            for j in N_:
                                try:
                                    q_final[i,j,ks] = q_rou[i,j]
                                    qe_final[i,j,ks] = qe_rou[i,j]
                                    w_final[i,j,ks] = w_rou[i,j]
                                    v_final[i,j,ks] = v_rou[i,j]
                                except:
                                    q_final[i,j,ks] = 0
                                    qe_final[i,j,ks] = 0
                                    w_final[i,j,ks] = 0
                                    v_final[i,j,ks] = 0
                        Theta_[ks] = Theta[ks] - max([q_rou[i,j] for i in N_ for j in N_])
                        Phi_[ks] = Phi[ks] - max([v_rou[i,j] for i in N_ for j in N_])
                        DictRoutes[ks], DictRoutesList[ks] = GetRoutingList(ks,depots[ks],N,w_final)
                        DictNodes[ks] = {'F' : F_, 'D' : D_, 'S': S_, 'C': C_}
                    except:
                        pass
                        ##print('ERROR IN REOPTI?IWING VEHICLE %s' % ks)
                else:
                    ##print('No more feasible changes for Vehicle %s' % kd)
                    flag_descend = False
            else:
                ##print('No more feasible changes for Vehicle %s' % kd)
                flag_feasible = False
    solution_swapkdks = {'DictRoutes' : DictRoutes,
                         'DictNodes' : DictNodes,
                         'DEMS_' : DEMS_,
                         'Q0_' : Q0_,
                         'q_final' : q_final,
                         'v_final' : v_final,
                         'w_final' : w_final,
                         'Phi_' : Phi_,
                         'Theta_' : Theta_,
                         'cdv' : cdv,
                         'cdw' : cdw,
                         'banlist' : banlist}            
    return solution_swapkdks

def AddNodeToSet(dictclass, add, F, D, S, C):
    cla = dictclass[add]
    if cla == 'F':
        F = F + [add]
    elif cla == 'D':
        D = D + [add]
    elif cla == 'S':
        S = S + [add]
    elif cla == 'C':
        C = C + [add]
    else:
        pass
    return F,D,S,C

def RemoveNodeFromSet(dictclass, rem, F, D, S, C):
    cla = dictclass[rem]
    if cla == 'F':
        F = [f for f in F if f != rem]
    elif cla == 'D':
        D = [d for d in D if d != rem]
    elif cla == 'S':
        S = [s for s in S if s != rem]
    elif cla == 'C':
        C = [c for c in C if c != rem]
    else:
        pass
    return F,D,S,C

def AddAndRemoveFromRoute(dictclass, DEMS_, P, DictNodes, k, add, DemAdd, rem, DemRem, r, depots, Q0_, Phi, Theta, nu, omega, rho, gamma):
    demms = DEMS_.copy()
    DEM_ = demms[k]
    Q0 = Q0_[k].copy()
    F_ = DictNodes[k]['F']
    D_ = DictNodes[k]['D']
    S_ = DictNodes[k]['S']
    C_ = DictNodes[k]['C']
    d0 = depots[k]
    ##print('AddAndRemoveFromRoute: Original demands ', DEM_)
    if add != None:
        ##print('AddAndRemoveFromRoute: Attempting to add node %s to vehicle %s' % (add,k))
        F_, D_, S_, C_ = AddNodeToSet(dictclass, add, F_, D_, S_, C_)
        DEM_[add] = [0 for p in P]
        for p in P:
            aux = DemAdd[p]
            DEM_[add][p] = aux # Demand for the new node
    if rem != None:
        ##print('AddAndRemoveFromRoute: Attempting to remove node %s to vehicle %s' % (rem,k))
        F_, D_, S_, C_ = RemoveNodeFromSet(dictclass, rem, F_, D_, S_, C_)
        for p in P:
            aux = DemRem[p]
            DEM_[rem][p] = DEM_[rem][p] - aux # If rem is depot, it will receive less feight
            # If rem is client, it will have demand 0
    N_ = F_ + D_ + S_ + C_
    q_rou, w_rou, v_rou = {},{},{}
    for i in N_:
        for j in N_:
            q_rou[i,j,k] = 0
            v_rou[i,j,k] = 0
            w_rou[i,j,k] = 0
    N_mind0 = [n for n in N_ if n != d0]
    for n in N_mind0:
        if max([np.absolute(DEM_[n][p]) for p in P]) < 1:
            ##print('AddAndRemoveFromRoute: Removing node %s from route of vehicle %s because of empty demand' % (n,k))
#             ##printx = True
            F_, D_, S_, C_ = RemoveNodeFromSet(dictclass, n, F_, D_, S_, C_)
    Route, RouteList = [], []
    NewDictNodes = {'F' : F_, 'D': D_, 'S': S_, 'C': C_}
    ##print('AddAndRemoveFromRoute: Vehicle %s, Nodes: ' % k, NewDictNodes)
    ##print('AddAndRemoveFromRoute: Vehicle %s, Demands: ' % k, DEM_)
    flag_optim = True
    N_ = F_ + D_ + S_ + C_
    if len(N_) > 2:
        data_routing = {'Q0' : Q0,
                        'DEM': DEMS_,
                        'N': N_,
                        'F' : F_,
                        'D': D_,
                        'S': S_,
                        'C': C_,
                        'P' : P,
                        'Phi' : Phi[k],
                        'Theta': Theta[k],
                        'nu' : nu,
                        'omega': omega, 
                        'd0' : depots[k],
                        'gamma': gamma[k],
                        'rho' : rho[k],
                        'r' : r}
        model_rou = FlowRouting(data_routing)
        model_rou.optimize()
        try:
            q, w, v = model_rou.__data
            q_rou2 = model_rou.getAttr('x', q)
            w_rou2 = model_rou.getAttr('x', w)
            v_rou2 = model_rou.getAttr('x', v)
            for (i,j) in q_rou2:
                q_rou[i,j,k] = q_rou2[i,j]
                w_rou[i,j,k] = w_rou2[i,j]
                v_rou[i,j,k] = v_rou2[i,j]
            Route, RouteList = GetRoutingList(k,d0,N_,w_rou)
            ##print('End routing vehicle %s' % k)
        except:
            flag_optim = False
            ##print('Infeasible routing for vehicle %s' % k)
    elif len(N_) == 2:
        j = [n for n in N_ if n != d0][0]
        w_rou[d0, j, k] = 1
        q_rou[d0, j, k] = sum([Q0[p]*omega[p] for p in P])
        v_rou[d0, j, k] = sum([Q0[p]*nu[p] for p in P])
        w_rou[j, d0, k] = 1
        q_rou[j, d0, k] = 0
        v_rou[j, d0, k] = 0
        Route, RouteList = [(d0,j), (j,d0)], [d0,j]
    else:
        pass
    CapVolRem = Theta[k] - max([q_rou[i,j,k] for i in N_ for j in N_])
    CapWeiRem = Phi[k] - max([v_rou[i,j,k] for i in N_ for j in N_])
    if flag_optim:
        FreightCost = sum([q_rou[i,j,k]*gamma[k]*r[i,j] for i in N_ for j in N_])
    else:
        FreightCost = np.infty
    return FreightCost, Route, RouteList, DEM_, Q0, CapVolRem, CapWeiRem, q_rou, w_rou, v_rou, NewDictNodes

def AuxRoutingKd(DEMS_, P, DictNodes, k, add, DemAdd, sat, r, depots, Q0_, Phi, Theta, nu, omega, rho, gamma, omegaeff):
    # Kd gets a new node on its route. This new node has demand and, as before
    # was served by a satellite, now that satelite receives less freight
    demms = DEMS_.copy()
    DEM_ = demms[k]
    Q0 = Q0_[k].copy()
    F_ = DictNodes[k]['F']
    D_ = DictNodes[k]['D']
    S_ = DictNodes[k]['S']
    C_ = DictNodes[k]['C']
    d0 = depots[k]
    if add != None:
        ##print('AuxRoutingKd: adding node %s to route of vehicle %s' % (add, k))
        C_ = C_ + [add]
        DEM_[add] = [0 for p in P]
        for p in P:
            aux = DemAdd[p]
            DEM_[sat][p] = DEM_[sat][p] - aux # satellite receives less freight
            DEM_[add][p] = aux # Demand for the new node
    N_ = F_ + D_ + S_ + C_
    q_rou, w_rou, v_rou = {},{},{}
    for i in N_:
        for j in N_:
            q_rou[i,j,k] = 0
            v_rou[i,j,k] = 0
            w_rou[i,j,k] = 0
    N_mind0 = [n for n in N_ if n != d0]
    for n in N_mind0:
        if max([np.absolute(DEM_[n][p]) for p in P]) < 1:
            ##print('AuxRoutingKd: Removing node %s from route of vehicle %s because of empty demand' % (n,k))
            F_ = [f for f in F_ if f != n]
            D_ = [d for d in D_ if d != n]
            S_ = [s for s in S_ if s != n]
            C_ = [c for c in C_ if c != n]
            N_ = F_ + D_ + S_ + C_
    Route, RouteList = [], []
    NewDictNodes = {'F' : F_, 'D': D_, 'S': S_, 'C': C_}
    ##print('AuxRoutingKd: Vehicle %s, Nodes: ' % k, NewDictNodes)
    # Consolidation of weight and volume
    flag_optim = True
    if len(N_) > 2:
        data_routing = {'Q0' : Q0,
                        'DEM': DEM_,
                        'N': N_,
                        'F' : F_,
                        'D': D_,
                        'S': S_,
                        'C': C_,
                        'P' : P,
                        'Phi' : Phi[k],
                        'Theta': Theta[k],
                        'nu' : nu,
                        'omega': omega,
                        'omegaeff': omegaeff,
                        'd0' : depots[k],
                        'gamma': gamma[k],
                        'rho' : rho[k],
                        'r' : r}
        model_rou = FlowRouting(data_routing)
        model_rou.optimize()
        try:
            q, w, v = model_rou.__data
            q_rou2 = model_rou.getAttr('x', q)
            w_rou2 = model_rou.getAttr('x', w)
            v_rou2 = model_rou.getAttr('x', v)
            for (i,j) in q_rou2:
                q_rou[i,j,k] = q_rou2[i,j]
                w_rou[i,j,k] = w_rou2[i,j]
                v_rou[i,j,k] = v_rou2[i,j]
            Route, RouteList = GetRoutingList(k,d0,N_,w_rou)
            ##print('End routing vehicle %s' % k)
        except:
            flag_optim = False
            ##print('Infeasible routing for vehicle %s' % k)
    elif len(N_) == 2:
        j = [n for n in N_ if n != d0][0]
        w_rou[d0, j, k] = 1
        q_rou[d0, j, k] = sum([Q0[p]*omega[p] for p in P])
        v_rou[d0, j, k] = sum([Q0[p]*nu[p] for p in P])
        w_rou[j, d0, k] = 1
        q_rou[j, d0, k] = 0
        v_rou[j, d0, k] = 0
        Route, RouteList = [(d0,j), (j,d0)], [d0,j]
    else:
        pass
    CapVolRem = Theta[k] - max([q_rou[i,j,k] for i in N_ for j in N_])
    CapWeiRem = Phi[k] - max([v_rou[i,j,k] for i in N_ for j in N_])
    if flag_optim:
        FreightCost = sum([q_rou[i,j,k]*gamma[k]*r[i,j] for i in N_ for j in N_])
    else:
        FreightCost = np.infty
    return FreightCost, Route, RouteList, DEM_, Q0, CapVolRem, CapWeiRem, q_rou, w_rou, v_rou, NewDictNodes
    

def AuxRoutingKs(DEMS_, P, DictNodes, k, ex, DemRem, sat, r, depots, Q0_, Phi, Theta, nu, omega, rho, gamma, omegaeff):
    # Ks got a node removed. So Q0 changes
    demms = DEMS_.copy()
    DEM_ = demms[k]
    Q0 = Q0_[k].copy()
    F_ = DictNodes[k]['F']
    D_ = DictNodes[k]['D']
    S_ = DictNodes[k]['S']
    C_ = DictNodes[k]['C']
    d0 = depots[k]
    N_ = F_ + D_ + S_ + C_
    C_ = [c for c in C_ if c != ex]
    DEM_[ex] = [0 for p in P]
    for p in P:
        aux = DemRem[p]
        Q0[p] = Q0[p] - aux # Vehicle starts with less freight
        DEM_[ex][p] = 0
    N_ = F_ + D_ + S_ + C_
    ##print('AuxRoutingKs: Vehicle %s, Nodes: ' % k, DictNodes[k])
    q_rou, w_rou, v_rou = {},{},{}
    for i in N_:
        for j in N_:
            q_rou[i,j,k] = 0
            v_rou[i,j,k] = 0
            w_rou[i,j,k] = 0
    Route, RouteList = [], []
    # Consolidation of weight and volume
    Weight = {}
    Volume = {}
    for i in N_:
        Weight[i] = sum([DEM_[i][p]*omega[p] for p in P])
        Volume[i] = sum([DEM_[i][p]*nu[p] for p in P])
    flag_optim = True
    if len(N_) > 2:
        data_routing = {'Q0' : Q0,
                        'DEM': DEMS_,
                        'N': N_,
                        'F' : F_,
                        'D': D_,
                        'S': S_,
                        'C': C_,
                        'P' : P,
                        'Phi' : Phi[k],
                        'Theta': Theta[k],
                        'nu' : nu,
                        'omega': omega,
                        'omegaeff': omegaeff,
                        'd0' : depots[k],
                        'gamma': gamma[k],
                        'rho' : rho[k],
                        'r' : r}
        model_rou = FlowRouting(data_routing)
        model_rou.optimize()
        model_rou.optimize()
        try:
            q, w, v = model_rou.__data
            q_rou2 = model_rou.getAttr('x', q)
            w_rou2 = model_rou.getAttr('x', w)
            v_rou2 = model_rou.getAttr('x', v)
            for (i,j) in q_rou2:
                q_rou[i,j,k] = q_rou2[i,j]
                w_rou[i,j,k] = w_rou2[i,j]
                v_rou[i,j,k] = v_rou2[i,j]
            Route, RouteList = GetRoutingList(k,d0,N_,w_rou)
            ##print('End routing vehicle %s' % k)
        except:
            flag_optim = False
            ##print('Infeasible routing for vehicle %s' % k)
    elif len(N_) == 2:
        j = [n for n in N_ if n != d0][0]
        w_rou[d0, j, k] = 1
        q_rou[d0, j, k] = sum([Q0[p]*omega[p] for p in P])
        v_rou[d0, j, k] = sum([Q0[p]*nu[p] for p in P])
        w_rou[j, d0, k] = 1
        q_rou[j, d0, k] = 0
        v_rou[j, d0, k] = 0
        Route, RouteList = [(d0,j), (j,d0)], [d0,j]
    else:
        pass
    CapVolRem = Theta[k] - max([q_rou[i,j,k] for i in N_ for j in N_])
    CapWeiRem = Phi[k] - max([v_rou[i,j,k] for i in N_ for j in N_])
    if flag_optim:
        FreightCost = sum([q_rou[i,j,k]*gamma[k]*r[i,j] for i in N_ for j in N_])
    else:
        FreightCost = np.infty
    NewDictNodes = {'F' : F_, 'D': D_, 'S': S_, 'C': C_}
    return FreightCost, Route, RouteList, DEM_, Q0, CapVolRem, CapWeiRem, q_rou, w_rou, v_rou, NewDictNodes

def ImproveOptimalSwapKsKd(RCVs, data, banlist, DictSatKd, cdv, cdw,  DictRoutes, DictRoutesList, DictNodes, DEMS_, Q0_, q_final, v_final, w_final, Phi_, Theta_, depots):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    V_i = data['V_i']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    omegaeff = data['omegaeff']
    rho = data['rho']
    gamma = data['gamma']
    DEM = data['DEM']
    r = data['r']
    A = data['A']
    N = F+D+S+C
      
    ##print('starting: ImproveOptimalSwapKsKd')
    # PARAMETER THAT SAYS HOW MANY VEHICLES FROM DEPOTS ARE BEING USED:
    depots_list = []
    for dep in depots.values():
        if dep not in depots_list:
            depots_list.append(dep)
    VehiclesPerDepot = {}
    for dep in depots_list:
        VehiclesPerDepot[dep] = sum([w_final[dep,j,k] for j in N for k in V_i[dep]])
    ##print(VehiclesPerDepot)
    for ks in RCVs.keys():
        ##print('Vehicle %s' % ks)
        flag_feasible = True
        flag_descend = True
        while flag_feasible and flag_descend:
            ex, q_ex, v_ex = GetNode2ExcludeFromVs(DictRoutes[ks], q_final, v_final,C, ks, gamma, r, banlist)
            if ex != None:
                # patch qex y vex (sometimes it has errors)
                q_ex = sum([DEM[ex][p]*omega[p] for p in P])
                v_ex = sum([DEM[ex][p]*nu[p] for p in P])
                ##print('Original demand of node %s: ' % ex, DEM[ex])
                ##print('Original freight of node %s: ' % ex, q_ex)
                sat = depots[ks]
                kd, rerouting_kd = MinRouteVariation([DictSatKd[sat]], ex, r, DictRoutes)
                # Backup for satellite demand
                dem_sat = [dem for dem in DEMS_[kd][sat]]
            else:
                sat = None
            # If there is a satelite...
            if sat != None and A[ex,kd] > 0:
                IncumbentCost = np.inf
                # PrevCost: routing cost for kd and ks without changes
                Costkd_pre = ComputeRouteCost(q_final, DictRoutes[kd], kd, gamma, r)
                Costks_pre = ComputeRouteCost(q_final, DictRoutes[ks], ks, gamma, r)
                PrevCost = Costkd_pre + Costks_pre
                ##print('Attempting to remove node %s from route of vehicle %s (sat = %s)' % (ex, ks,sat))
#                Aux##printPreRerouting(DEMS_, Q0_, kd , nu, omega, P, DictNodes)
                Sol_kd = AuxRoutingKd(DEMS_, P, DictNodes, kd, ex, DEM[ex], sat, r, depots, Q0_, Phi, Theta, nu, omega, rho, gamma, omegaeff)
                Costkd_pos = Sol_kd[0]
#                Aux##printPreRerouting(DEMS_, Q0_, ks , nu, omega, P, DictNodes)
                Sol_ks = AuxRoutingKs(DEMS_, P, DictNodes, ks, ex, DEM[ex], sat, r, depots, Q0_, Phi, Theta, nu, omega, rho, gamma, omegaeff)
                Costks_pos = Sol_ks[0]
                # CHECK IF SATELLITE HAS EMPTY ROUTE
                if Sol_ks[10]['C']:
                    IncumbentCost = Costkd_pos + Costks_pos
                else:
                    ##print('Vehicle %s has an empty route' % ks)
                    if VehiclesPerDepot[depots[ks]] - 1 == 0:
                        IncumbentCost = Costkd_pos + Costks_pos - 1000
                        ##print('Attempting to close satelite %s' % depots[ks])
                ##print('Incumbent: ', IncumbentCost, ' previous: ', PrevCost)
                if IncumbentCost <= PrevCost:
                    ##print('Updating routes for vehicles kd = %s and ks = %s' % (kd,ks))
                    DictSol = {kd : Sol_kd, ks: Sol_ks}
                    #FreightCost, Route, RouteList, DEM, Q0, CapVolRem, CapWeiRem, q_rou, w_rou, v_rou
                    for k in [kd,ks]:
                        OldRoute = DictRoutes[k]
                        for (i,j) in OldRoute:
                            q_final[i,j,k] = 0
                            w_final[i,j,k] = 0
                            v_final[i,j,k] = 0
                        DictRoutes[k] = DictSol[k][1]
                        DictRoutesList[k] = DictSol[k][2]
                        DEMS_[k] = DictSol[k][3]
                        Q0_[k] = DictSol[k][4]
                        Phi_[k] = DictSol[k][5]
                        Theta_[k] = DictSol[k][6]
                        for (i,j) in DictSol[k][1]:
                            q_final[i,j,k] = DictSol[k][7][i,j,k]
                            w_final[i,j,k] = DictSol[k][8][i,j,k]
                            v_final[i,j,k] = DictSol[k][9][i,j,k]
                        # Nodes are modified
                        DictNodes[k] = DictSol[k][10]
                    # Se agrega nuevo Node a kd y se quita de ks:
                    # Remaining capacities of depots are modified:
                    cdw[depots[ks]] = cdw[depots[ks]] + q_ex
                    cdv[depots[ks]] = cdv[depots[ks]] + v_ex
                    if Sol_ks[10]['C']:
                        pass
                    else:
                        VehiclesPerDepot[depots[ks]] = VehiclesPerDepot[depots[ks]] - 1
                    ##print('There was an exchange between kd = %s y ks = %s' % (kd,ks))
                else:
                    ##print('There was not an exchange between kd = %s y ks = %s' % (kd,ks))
                    DEMS_[kd][sat] = dem_sat
                    del DEMS_[kd][ex]
                    flag_descend = False
            else:
                ##print('No more feasible changes for Vehicle %s' % ks)
                flag_feasible = False
    solution_swapkskd = {'DictRoutes' : DictRoutes,
                         'DictNodes' : DictNodes,
                         'DEMS_' : DEMS_,
                         'Q0_' : Q0_,
                         'q_final' : q_final,
                         'v_final' : v_final,
                         'w_final' : w_final,
                         'Phi_' : Phi_,
                         'Theta_' : Theta_,
                         'cdv' : cdv,
                         'cdw' : cdw,
                         'banlist' : banlist}            
    return solution_swapkskd

def Steps1To3(data):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']
    DEM = data['DEM']
    Lambd = data['Lambd']
    Omega = data['Omega']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    omegaeff = data['omegaeff']
    rho = data['rho']
    delta = data['delta']
    gamma = data['gamma']
    r = data['r']
    epsil = data['epsil']
    A = data['A']
    firmswithout = F[- max(int(len(F)*0.2), 1):] #This is just for saying that 20% of the firms (or at least 1)
    # don't have vehicles
    Fw = [f for f in firmswithout]
    FminFw = [f for f in F if f not in Fw]
    Vs = [item for sublist in [V_i[s] for s in S] for item in sublist]
    Vd = [item for sublist in [V_i[d] for d in D] for item in sublist]
    N = F+D+S+C
    DcupS = D + S
    ScupC = S + C
    NminC = F + D + S
    # Other parameters
    VolClient = {}
    WeightClient = {}
    for client, dem in DEM.items():
        VolClient[client] = sum([dem[p]*nu[p] for p in P])
        WeightClient[client] = sum([dem[p]*omega[p] for p in P])

    MinVolDep = {}
    MinWeightDep = {}
    for i in DcupS:
        MinVolDep[i] = min(Lambd[i], sum([Phi[k] for k in V_i[i]]))
        MinWeightDep[i] = min(Omega[i], sum([Theta[k] for k in V_i[i]]))
    ServCost = []
    PfromFw = [item for sublist in [P_f[f] for f in Fw] for item in sublist]
    F_p = {}
    for f in F:
        for p in P_f[f]:
            F_p[p] = f
    # Serving cost for delivering to customers
    for i in D:
#         gamma_ki = max([gamma[v] for v in V_i[i]])
#         rho_ki = max([rho[v] for v in V_i[i]])
        for c in C:
            Weight_cf = {}
            gamma_kf = {}
            rho_kf = {}
            for f in F:
                Weight_cf[f] = sum([DEM[c][p]*omegaeff[p] for p in P_f[f]])
                gamma_kf[f] = 0
                rho_kf[f] = 0
#                 gamma_kf[f] = max([gamma[v] for v in V_i[f]])
#                 rho_kf[f] = max([rho[v] for v in V_i[f]])
            # Check if customer c demanded products from firms without vehicles
            if sum([DEM[c][p] for p in PfromFw]) > 0:
                flag_fw = True
                FirmsToVisit = [f for f in Fw if max([DEM[c][p] for p in P_f[f]]) > 0]
            else:
                flag_fw = False
            load_cost_f1 = sum([r[f,i]*(gamma_kf[f]*Weight_cf[f] + rho_kf[f]) for f in FminFw])
            for k in V_i[i]:
                if A[c,k] > 0:
                    gamma_ki = gamma[k]
                    rho_ki = rho[k]
                    if flag_fw:
                        f0, load_cost_f2 = GreedyRoutingForServingCost(i,
                                                                   sum([Weight_cf[f] for f in FminFw]),
                                                                   FirmsToVisit,
                                                                   Weight_cf,
                                                                   gamma_ki,
                                                                   rho_ki,
                                                                   r)
                        del_cost = r[f0,c]*(gamma_ki*sum([Weight_cf[f] for f in Fw]) + rho_ki)
                    else:
                        load_cost_f2 = 0
                        del_cost = r[i,c]*(gamma_ki*sum([Weight_cf[f] for f in F]) + rho_ki)
                    """HERE WE CAN ADD ADDITIONAL COSTS FOR ROUTING"""
                    sc = load_cost_f1 + load_cost_f2 + del_cost + delta[k]
                    ServCost.append([i,c,k, sc, VolClient[c], WeightClient[c]])
    # serving cost for satellite
    for i in S:
        for c in C:
            Weight_cf = {}
            gamma_kf = {}
            rho_kf = {}
            for f in F:
                Weight_cf[f] = sum([DEM[c][p]*omegaeff[p] for p in P_f[f]])
                gamma_kf[f] = 0
                rho_kf[f] = 0
#                 gamma_kf[f] = max([gamma[v] for v in V_i[f]])
#                 rho_kf[f] = max([rho[v] for v in V_i[f]])
            # Check if customer c demanded products from firms without vehicles
            if sum([DEM[c][p] for p in PfromFw]) > 0:
                flag_fw = True
                FirmsToVisit = [f for f in Fw if max([DEM[c][p] for p in P_f[f]]) > 0]
            else:
                flag_fw = False
            load_cost_f1 = GetMinimalLoadCostF1(r, i, gamma, Weight_cf, rho, FminFw, D, V_i)
            if flag_fw:
                del_cost = GetBestDeliveringCost(r, i, gamma, Weight_cf, rho, FirmsToVisit, D, V_i)
            else:
                del_cost = 0
            for k in V_i[i]:
                if A[c,k] > 0:
                    gamma_ki = max([gamma[v] for v in V_i[i]])
                    rho_ki = max([rho[v] for v in V_i[i]])
                    del_cost = r[i,c]*(gamma_ki*sum([Weight_cf[f] for f in Fw]) + rho_ki) + epsil[i]
                    sc = load_cost_f1 + del_cost + epsil[i] + delta[k]
                    ServCost.append([i,c,k,sc, VolClient[c], WeightClient[c]])
    df_sc = pd.DataFrame(data = ServCost, columns = ['depot','customer','vehicle','servcost','volume','weight'])
    df_sc = df_sc.sort_values(by='servcost', ascending = True).reset_index(drop=True)
    openedS = []
    usedK = []
    DEM_i = {} # Dictionary for depots demands
    for i in DcupS:
        DEM_i[i] = [0 for p in P]
    h_opt = {}
    m_opt = {}
    x_opt = {}
    y_opt = {}
    u_opt = {}
    z_opt = {}
    Weight_i = {}
    Volume_i = {}
    for i in DcupS:
        Weight_i[i] = 0
        Volume_i[i] = 0
    Weight_k = {}
    Volume_k = {}
    for k in K:
        Weight_k[k] = 0
        Volume_k[k] = 0
#     #print(df_sc)
    while df_sc.shape[0] > 0:
        # Always check first element in dataframe
        i = int(df_sc.loc[0]['depot'])
        c = int(df_sc.loc[0]['customer'])
        k = int(df_sc.loc[0]['vehicle'])
        w = df_sc.loc[0]['weight']
        v = df_sc.loc[0]['volume']
#        #print(df_sc.head())
#        #print(df_sc.shape[0])
        #print('Customer %s trying to be added to depot %s' %(c,i))
#        #print('Depot incumbent weight: %s of %s' % (Weight_i[i] + w,  MinWeightDep[i]))
#        #print('Depot incumbent Volume: %s of %s' % (Volume_i[i] + v,  MinVolDep[i]))
        if Weight_i[i] + w <= MinWeightDep[i] and Volume_i[i] + v <= MinVolDep[i]:
#            #print('Vehicle incumbent weight: %s of %s' % (Weight_k[k] + w,  Theta[k]))
#            #print('Vehicle incumbent Volume: %s of %s' % (Volume_k[k] + v,  Phi[k]))
            if Weight_k[k] + w <= Theta[k] and Volume_k[k] + v <= Phi[k]:
            # Add
                for p in P:
                    if DEM[c][p] > 0:
                        h_opt[p,i,c] = DEM[c][p]
                        m_opt[p,c,k] = DEM[c][p]
                        DEM_i[i][p] = DEM_i[i][p] + DEM[c][p]
                        fp = F_p[p]
                        if fp in Fw and k in Vd:
                            if (p,fp,k) in x_opt.keys():
                                x_opt[p,fp,k] = x_opt[p,fp,k] + DEM[c][p]
                            else:
                                x_opt[p,fp,k] = DEM[c][p]
                Weight_i[i] = Weight_i[i] + w
                Volume_i[i] = Volume_i[i] + v
                Weight_k[k] = Weight_k[k] + w
                Volume_k[k] = Volume_k[k] + v
                z_opt[k,c] = 1
                # Delete customer from set (becasue it was assigned)
                df_sc = df_sc[df_sc['customer'] != c]
                if i in S and i not in openedS:
                    openedS.append(i)
                    u_opt[i] = 1
                    # Substract the opening cost
                    df_sc['servcost'] = np.where(df_sc['depot'] == i,
                                                 df_sc['servcost'] - epsil[i],
                                                 df_sc['servcost'])
                if k not in usedK:
                    usedK.append(k)
                    y_opt[k] = 1
                    # Substract the opening cost
                    df_sc['servcost'] = np.where(df_sc['vehicle'] == k,
                                                 df_sc['servcost'] - delta[k],
                                                 df_sc['servcost'])
#                #print('Customer %s added to depot %s' %(c,i))
            else:
                df_sc = df_sc[1:]
        else:
            df_sc = df_sc[1:]
#         wm = df_sc['weight'].min()
#         vm = df_sc['volume'].min()
#         if Weight_i[i] == MinWeightDep[i] or Volume_i[i] == MinVolDep[i]:
#             df_sc = df_sc[df_sc['depot'] != i]
#         if Weight_k[k] == Theta[k] or Volume_k[k] == Phi[k]:
#             df_sc = df_sc[df_sc['vehicle'] != k]
        # Reorder by  servingcost
        df_sc = df_sc.sort_values(by='servcost', ascending = True).reset_index(drop=True)
    # Now, we know the satellites' demand for products. So, we will assign dustributor(s) to
    # each satellite as if they were customers
    # Serving cost for products from firms with vehicles
#     for s in openedS:
#         #print(DEM_i[s])
#     #print('Opened satellites = %s' % len(openedS))
    ServCost = []
    for f in FminFw:
        for s in openedS:
            for p in P_f[f]:
                if DEM_i[s][p] > 0:
                    w = DEM_i[s][p]*omega[p]
                    we = DEM_i[s][p]*omegaeff[p]
                    v = DEM_i[s][p]*nu[p]
#                     for k in V_i[f]:
#                         gamma_kf = gamma[k]
#                         rho_kf = rho[k]
#                         sc = r[f,s]*(gamma_kf*w + rho_kf)
#                         ServCost.append([f, s, p, k, sc, v, w])
#                     gamma_kf = max([gamma[v] for v in V_i[f]])
#                     rho_kf = max([rho[v] for v in V_i[f]])
                    gamma_kf = 0
                    rho_kf = 0
                    for d in D:
                        for k in V_i[d]:
                            gamma_kd = gamma[k]
                            rho_kd = rho[k]
                            sc = r[f,d]*(gamma_kf*we + rho_kf) + r[d,s]*(gamma_kd*we + rho_kd)
                            ServCost.append([d, s, p, k, sc, v, w])
    # Serving cost for products from firms without vehicles:
    for f in Fw:
        for s in openedS:
            for p in P_f[f]:
                if DEM_i[s][p] > 0:
                    w = DEM_i[s][p]*omega[p]
                    we = DEM_i[s][p]*omegaeff[p]
                    v = DEM_i[s][p]*nu[p]
                    for d in D:
                        for k in V_i[d]:
                            gamma_kd = gamma[k]
                            rho_kd = rho[k]
                            sc = r[f,d]*(gamma_kd*we + rho_kd) + r[d,s]*(gamma_kd*we + rho_kd)
                            if k not in usedK:
                                sc = sc + delta[k]
                            ServCost.append([d, s, p, k, sc, v, w])

    df_sc = pd.DataFrame(data = ServCost, columns = ['depot','satellite','product','vehicle','servcost','volume','weight'])
    df_sc = df_sc.sort_values(by='servcost', ascending = True).reset_index(drop=True)
    df_sc['fixcostvehicle'] = [delta[v] for v in df_sc['vehicle'].tolist()]
    df_sc['servcost'] = np.where(df_sc['vehicle'].isin(usedK), df_sc['servcost'], df_sc['servcost'] + df_sc['fixcostvehicle'])
    while df_sc.shape[0] > 0:
        # Always check first element in dataframe
        i = int(df_sc.loc[0]['depot'])
        s = int(df_sc.loc[0]['satellite'])
        p = int(df_sc.loc[0]['product'])
        k = int(df_sc.loc[0]['vehicle'])
        w = int(df_sc.loc[0]['weight'])
        v = int(df_sc.loc[0]['volume'])
        if i in F:
            condition1 = True
        else:
            condition1 = Weight_i[i] + w <= MinWeightDep[i] and Volume_i[i] + v <= MinVolDep[i]
            # Add
        condition2 = Weight_k[k] + w <= Theta[k] and Volume_k[k] + v <= Phi[k]
        if condition1 and condition2:
#            if DEM_i[s][p] == 0:
                #print('Warning: s = %s and p = %s' % (s,p))
            fp = F_p[p]
            h_opt[p,i,s] = DEM_i[s][p]
            # PATCH FOR MAKING PRODUCTS FROM FIRMS WITH VEHICLES APPEAR
            if fp not in Fw:
                m_opt[p,s,k] = 0
            else:
                m_opt[p,s,k] = DEM_i[s][p]
            Weight_k[k] = Weight_k[k] + w
            Volume_k[k] = Volume_k[k] + v
            if i in D:
                DEM_i[i][p] = DEM_i[i][p] + DEM_i[s][p]
                Weight_i[i] = Weight_i[i] + w
                Volume_i[i] = Volume_i[i] + v
            if fp in Fw and k in Vd:
                if (p,fp,k) in x_opt.keys():
                    x_opt[p,fp,k] = x_opt[p,fp,k] + DEM_i[s][p]
                else:
                    x_opt[p,fp,k] = DEM_i[s][p]
            if k not in usedK:
                usedK.append(k)
                y_opt[k] = 1
                df_sc['servcost'] = df_sc['servcost'] - delta[k]
            DEM_i[s][p] = 0
            df_sc = df_sc[1:]
            df_sc = df_sc[~((df_sc['satellite'] == s) & (df_sc['product'] == p))]
        else:
            df_sc = df_sc[1:]
        wm = df_sc['weight'].min()
        vm = df_sc['volume'].min()
        if i in D:
            if Weight_i[i] + wm > MinWeightDep[i] or Volume_i[i] + vm > MinVolDep[i]:
                df_sc = df_sc[df_sc['depot'] != i]
        if Weight_k[k] + wm > Theta[k] or Volume_k[k] + vm > Phi[k]:
            df_sc = df_sc[df_sc['vehicle'] != k]
        # Delete customer from set (becasue it was assigned)
        if sum([DEM_i[s][p_] for p_ in P]) < 1:
            df_sc = df_sc[df_sc['satellite'] != s]
        # Reorder by  servingcost
        df_sc = df_sc.sort_values(by='servcost', ascending = True).reset_index(drop=True)
    cdv = {}
    cdw = {}
    for i in DcupS:
        cdv[i] = Lambd[i] - Volume_i[i]
        cdw[i] = Omega[i] - Weight_i[i]
    return m_opt, u_opt, x_opt, y_opt, z_opt, cdv, cdw
    

def ExecuteMultiEchelon(data, filename = None, preplots = False):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']
    DEM = data['DEM']
    Lambd = data['Lambd']
    Omega = data['Omega']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    omegaeff = data['omegaeff']
    rho = data['rho']
    delta = data['delta']
    gamma = data['gamma']
    epsil = data['epsil']
    r = data['r']
    N = F+D+S+C
    dictclass = {}
    for f in F:
        dictclass[f] = 'F'
    for d in D:
        dictclass[d] = 'D'
    for s in S:
        dictclass[s] = 'S'
    for c in C:
        dictclass[c] = 'C'
    m_opt, u_opt, x_opt, y_opt, z_opt, cdv, cdw = Steps1To3(data)

    solution = MultiEchelonRouting(data, x_opt, y_opt, m_opt, z_opt, u_opt)
    # Unpack data from solution
    q_final = solution['q_final']
    qe_final = solution['qe_final']
    v_final = solution['v_final']
    w_final = solution['w_final']
    y_final = {}
    for k in K:
        try:
            y_final[k] = min(sum([w_final[i,j,k] for i in N for j in N]), 1)
        except:
            y_final[k] = 0
    DictRoutes = solution['DictRoutes']
    DictRoutesList = solution['DictRoutesList']
    DictNodes = solution['DictNodes']
    Theta_ = solution['Theta_']
    Phi_ = solution['Phi_']
    Q0_ = solution['Q0_']
    DEMS_ = solution['DEMS_']
    """RETRIEVE ORIGINAL OBJECTIVE FUNCTION VALUE"""
    # Aux
    m_final = {}
    # Patch m_final:
    for i in N:
        for j in N:
            for k in K:
                m_final[i,j,k] = m_opt.get((i,j,k),0)
    # Patch w_final, q final, v_final:
    for i in N:
        for j in N:
            for k in K:
                if (i,j,k) not in q_final:
                    q_final[i,j,k] = 0
                if (i,j,k) not in w_final:
                    w_final[i,j,k] = 0
                if (i,j,k) not in v_final:
                    v_final[i,j,k] = 0
    if preplots:
        AuxSubPlot(data, w_final, figsize = (5,5), save = True, filename = filename)
    u_final = {}
    for s in S:
        u_final[s] = u_opt.get(s,0)
    # Cost of Satellites
    SatCost = sum([u_final[s]*epsil[s] for s in S])
    VehicCost = sum([y_final[k]*delta[k] for k in K])
    ArcCost = sum([w_final[i,j,k]*r[i,j]*rho[k] for i in N for j in N for k in K])
    FreightCost = sum([gamma[k]*qe_final[i,j,k]*r[i,j] for i in N for j in N for k in K])

    Opt = SatCost + VehicCost + ArcCost + FreightCost

    depots = {}
    for i in D+S:
        for k in V_i[i]:
            depots[k] = i
    ##print('LOOP: START!')
    # WORKING HERE
    CurrentOpt = Opt
    Vd = [item for sublist in [V_i[d] for d in D] for item in sublist]
    n_iters = 3
    iters = 1
    tries = 1
    while iters <= n_iters and tries < 2:
        ##print('Iter: %s, Try: %s' % (iters,tries))
        RCVd = GetMaxRoutingCosts(N, Vd, depots, DictNodes, r, gamma, w_final, q_final)
        # ADD FUNCTION FOR SORT RCVd dictionary by value
        ##print('PERMUTACIONES KD A KS')        
        solution_swapkdks = ImproveOptimalSwapKdKs(RCVd,
                                                   data,
                                                   cdv,
                                                   cdw, 
                                                   DictRoutes,
                                                   DictRoutesList,
                                                   DictNodes,
                                                   DEMS_,
                                                   Q0_,
                                                   q_final,
                                                   qe_final,
                                                   v_final,
                                                   w_final,
                                                   Phi_,
                                                   Theta_,
                                                   depots)
        # Unpack data
        DictRoutes = solution_swapkdks['DictRoutes']
        DictNodes = solution_swapkdks['DictNodes']
        DEMS_ = solution_swapkdks['DEMS_']
        Q0_ = solution_swapkdks['Q0_']
        q_final = solution_swapkdks['q_final']
        v_final = solution_swapkdks['v_final']
        w_final = solution_swapkdks['w_final']
        Phi_ = solution_swapkdks['Phi_']
        Theta_ = solution_swapkdks['Theta_']
        cdv = solution_swapkdks['cdv']
        cdw = solution_swapkdks['cdw']
        banlist = solution_swapkdks['banlist']
        # Patch w_final, q final, v_final:
        for i in N:
            for j in N:
                for k in K:
                    if (i,j,k) not in q_final:
                        q_final[i,j,k] = 0
                    if (i,j,k) not in w_final:
                        w_final[i,j,k] = 0
                    if (i,j,k) not in v_final:
                        v_final[i,j,k] = 0
        # Get Dictionary with this structure: key = satellite, value = vehicle from D that visits that satellite
        KminV_is = {}
        for s in S:
            KminV_is[s] = [k for k in K if k not in V_i[s]]
        Sserv1 = [s for s in S if sum([w_final[s,j,k] for j in N for k in KminV_is[s]]) == 1]
        DictSatKd = {}
        for kd in Vd:
            if DictNodes[kd]['S']:
                for s in DictNodes[kd]['S']:
                    if s in Sserv1:
                        DictSatKd[s] = kd
        Vs1 = [item for sublist in [V_i[s] for s in DictSatKd.keys()] for item in sublist]
        RCVs = GetMaxRoutingCosts(N, Vs1, depots, DictNodes, r, gamma, w_final, q_final)
        solution_swapkskd = ImproveOptimalSwapKsKd(RCVs,
                                                   data,
                                                   banlist,
                                                   DictSatKd,
                                                   cdv,
                                                   cdw, 
                                                   DictRoutes,
                                                   DictRoutesList,
                                                   DictNodes,
                                                   DEMS_,
                                                   Q0_,
                                                   q_final,
                                                   v_final,
                                                   w_final,
                                                   Phi_,
                                                   Theta_,
                                                   depots)
        DictRoutes = solution_swapkskd['DictRoutes']
        DictNodes = solution_swapkskd['DictNodes']
        DEMS_ = solution_swapkskd['DEMS_']
        Q0_ = solution_swapkskd['Q0_']
        q_final = solution_swapkskd['q_final']
        v_final = solution_swapkskd['v_final']
        w_final = solution_swapkskd['w_final']
        Phi_ = solution_swapkskd['Phi_']
        Theta_ = solution_swapkskd['Theta_']
        cdv = solution_swapkskd['cdv']
        cdw = solution_swapkskd['cdw']
        banlist = solution_swapkskd['banlist']
        # Patch w_final, q final, v_final:
        for i in N:
            for j in N:
                for k in K:
                    if (i,j,k) not in q_final:
                        q_final[i,j,k] = 0
                    if (i,j,k) not in w_final:
                        w_final[i,j,k] = 0
                    if (i,j,k) not in v_final:
                        v_final[i,j,k] = 0
        for s in S:
            if sum(w_final[s,j,k] for j in N for k in V_i[s]) < 1:
                u_final[s] = 0
            else:
                u_final[s] = 1
        for k in K:
            y_final[k] = max(min(sum([w_final[i,j,k] for i in N for j in N]),1),0)
        SatCost = sum([u_final[s]*epsil[s] for s in S])
        VehicCost = sum([y_final[k]*delta[k] for k in K])
        ArcCost = sum([w_final[i,j,k]*r[i,j]*rho[k] for i in N for j in N for k in K])
        FreightCost = sum([gamma[k]*q_final[i,j,k]*r[i,j] for i in N for j in N for k in K])
        Opt = SatCost + VehicCost + ArcCost + FreightCost
        
        ### STEP FOR VEHICLES FROM FIRMS ###
        
        
        iters = iters + 1
        if Opt < CurrentOpt:
            ##print('####################### REPORT FOR ITER %s #######################' % iters)
            ##print('Number of satellites open: %s at cost %s' % (sum([u_final[s] for s in S]), SatCost))
            ##print('Number of vehicles used: %s at cost %s' % (sum([y_final[k] for k in K]), VehicCost))
            ##print('Arc cost: %s' % ArcCost)
            ##print('Freight cost: %s' % FreightCost)
            ##print('Optimal value for original O.F: %s' % Opt)
            CurrentOpt = Opt
            tries = 1
        else:
            tries = tries + 1

    ##print('####################### FINAL REPORT #######################')
    for k in K:
        y_final[k] = max(min(sum([w_final[i,j,k] for i in N for j in N]),1),0)
    SatCost = sum([u_final[s]*epsil[s] for s in S])
    VehicCost = sum([y_final[k]*delta[k] for k in K])
    ArcCost = sum([w_final[i,j,k]*r[i,j]*rho[k] for i in N for j in N for k in K])
    FreightCost = sum([gamma[k]*qe_final[i,j,k]*r[i,j] for i in N for j in N for k in K])
    #print('Number of satellites open: %s at cost %s' % (sum([u_final[s] for s in S]), SatCost))
    #print('Number of vehicles used: %s at cost %s' % (sum([y_final[k] for k in K]), VehicCost))
    #print('Arc cost: %s' % ArcCost)
    #print('Freight cost: %s' % FreightCost)
    Opt = SatCost + VehicCost + ArcCost + FreightCost
    #print('Optimal value for original O.F: %s' % Opt)
    return q_final, w_final, u_final, y_final, DictRoutes, Opt

"""
FUNCTIONS FOR PLOTTING
"""

def PlotNodes(XY, F, D, S, C, figsize = (20,20)):
    fig, ax = plt.subplots(figsize= figsize)
    plt.scatter(XY[F,0],XY[F,1],color='red', label= 'Goods')
    plt.scatter(XY[D,0], XY[D,1],color='blue', label = 'Delivery')
    plt.scatter(XY[S,0], XY[S,1],color='green', label = 'Satellites') 
    plt.scatter(XY[C,0], XY[C,1],color='brown', label = 'Clients')
    for i in S:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in C:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    return fig, ax

def PlotAssignsSatCli(XY,F, D, S, C, model, figsize = (20,20)):
    l, u = model.__data
    N = F+D+S+C
    DcupS = D + S
    FcupD = D + S
    NminC = F + D + S
    l_opt = model.getAttr('x', l)
    u_opt = model.getAttr('x', u)
    colors = {}
    for s in NminC:
        colors[s] = tuple(np.random.rand(3))
    dictveh = {}
    fig, ax = plt.subplots(figsize = figsize)
    S_op = []
    for s in S:
        if u_opt[s] > 0:
            S_op.append(s)
    ##print(S_op)
    plt.scatter(XY[F,0],XY[F,1],color='red', label= 'Goods')
    plt.scatter(XY[D,0], XY[D,1],color='blue', label = 'Delivery')
    plt.scatter(XY[S_op,0], XY[S_op,1],color='green', label = 'Satellites') 
    plt.scatter(XY[C,0], XY[C,1],color='brown', label = 'Clients')
    for s in NminC:
        flag_v = True
        for c in C:
            if l_opt[s,c] > 0:
                x1, x2 = XY[s,0], XY[c,0]
                y1, y2 = XY[s,1], XY[c,1]
                plt.plot([x1,x2],[y1,y2],
                         color = colors[s],
                         linestyle = 'dashed',
                         label = 'Satelite %s' % s if flag_v else "")
                flag_v = False
    for i in F:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in D:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in S:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in C:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    plt.legend()
    plt.show()
    
def PlotAssignsVehCli(XY,F, D, S, C, V_i, model, figsize = (20,20)):
    y, z = model.__data
    Vs = [item for sublist in [V_i[s] for s in S] for item in sublist]
    Vd = [item for sublist in [V_i[d] for d in D] for item in sublist]
    Vf = [item for sublist in [V_i[f] for f in F] for item in sublist]
    VdcupVs = Vd + Vs
    VfcupVs = Vf + Vd
    K = Vf + Vd + Vs
    N = F+D+S+C
    DcupS = D + S
    FcupD = D + S
    NminC = F + D + S
    z_opt = model.getAttr('x', z)
    colors = {}
    for s in NminC:
        for k in V_i[s]:
            colors[k] = tuple(np.random.rand(3))
    dictveh = {}
    for i in NminC:
        for k in V_i[i]:
            dictveh[k] = i
    fig, ax = plt.subplots(figsize = figsize)
    plt.scatter(XY[F,0],XY[F,1],color='red', label= 'Goods')
    plt.scatter(XY[D,0], XY[D,1],color='blue', label = 'Delivery')
    plt.scatter(XY[S,0], XY[S,1],color='green', label = 'Satellites') 
    plt.scatter(XY[C,0], XY[C,1],color='brown', label = 'Clients')
    for k in K:
        flag_v = True
        for c in C:
            try:
                if z_opt[k,c] > 0:
                    s = dictveh[k]
                    x1, x2 = XY[s,0], XY[c,0]
                    y1, y2 = XY[s,1], XY[c,1]
                    plt.plot([x1,x2],[y1,y2],
                             color = colors[k],
                             linestyle = 'dashed',
                             label = 'Vehicle %s (%s)' % (k, dictveh[k]) if flag_v else "")
                    flag_v = False
            except:
                pass
    for i in F:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in D:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in S:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    for i in C:
        ax.annotate(i, (XY[i,0], XY[i,1]))
    plt.legend()
    plt.show()

def AuxSubPlot(data, w_opt, figsize = (20,20), save = False, filename = 'test'):
    # Unpacking data
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']
    DEM = data['DEM']
    Lambd = data['Lambd']
    Omega = data['Omega']
    Phi = data['Phi']
    Theta = data['Theta']
    nu = data['nu']
    omega = data['omega']
    rho = data['rho']
    delta = data['delta']
    gamma = data['gamma']
    r = data['r']
    X,Y = XY[:,0], XY[:,1]
    label = ['Goods' for f in F] + ['Delivery' for d in D] + ['Satellite' for s in S] + ['Clients' for c in C]
    n_label = [0 for f in F] + [1 for d in D] + [2 for s in S] + [3 for c in C]
    colors_xy = ['red','blue','green','brown']
    N = F + D + S + C
    NminC = F + D + S
    dictveh = {}
    for i in NminC:
        for k in V_i[i]:
            dictveh[k] = i
    K = [item for sublist in [V_i[i] for i in NminC] for item in sublist]
    cmapp = cm.get_cmap('viridis', len(K))
    colors = {}
    for k in K:
        if k % 2 == 0:
            colors[k] = cmapp(k)
        else:
            colors[k] = cmapp(K[::-1][k])
    plt.figure(figsize=figsize)
    plt.scatter(X[F], Y[F], label = 'Firms', color = 'red')
    plt.scatter(X[D], Y[D], label = 'Delivery', color = 'blue')
    plt.scatter(X[S], Y[S], label = 'Satellites', color = 'green')
    plt.scatter(X[C], Y[C], label = 'Clients', color = 'brown')
    for i in range(XY.shape[0]):
        x = X[i]
        y = Y[i]
        plt.text(x+0.3, y+0.3, i, fontsize=9)
    for f in F:
        for k in V_i[f]:
            for i in N:
                for j in N:
                    key = (i,j,k)
                    if key in w_opt:
                        if w_opt[key] > 0:
                            x1, x2 = XY[i,0], XY[j,0]
                            y1, y2 = XY[i,1], XY[j,1]
                            plt.plot([x1,x2],[y1,y2],
                                     color = colors[k],
                                     linestyle = 'dashed',
                                     label = 'Vehicle %s (%s)' % (k, dictveh[k]) if i == dictveh[k] else "")
    plt.legend()
    plt.title('Vehicles from Firms')
    if save:
        plt.tight_layout()
        plt.savefig('%s-firms.png' % filename, dpi = 250)
    
    plt.figure(figsize=figsize)
    plt.scatter(X[F], Y[F], label = 'Firms', color = 'red')
    plt.scatter(X[D], Y[D], label = 'Delivery', color = 'blue')
    plt.scatter(X[S], Y[S], label = 'Satellites', color = 'green')
    plt.scatter(X[C], Y[C], label = 'Clients', color = 'brown')
    for i in range(XY.shape[0]):
        x = X[i]
        y = Y[i]
        plt.text(x+0.3, y+0.3, i, fontsize=9)
    for d in D:
        for k in V_i[d]:
            for i in N:
                for j in N:
                    key = (i,j,k)
                    if key in w_opt:
                        if w_opt[key] > 0:
                            x1, x2 = XY[i,0], XY[j,0]
                            y1, y2 = XY[i,1], XY[j,1]
                            plt.plot([x1,x2],[y1,y2],
                                     color = colors[k],
                                     linestyle = 'dashed',
                                     label = 'Vehicle %s (%s)' % (k, dictveh[k]) if i == dictveh[k] else "")
    plt.legend()
    plt.title('Vehicles from Delivery')
    if save:
        plt.tight_layout()
        plt.savefig('%s-delivery.png' % filename, dpi = 250)
    
    plt.figure(figsize=figsize)
    plt.scatter(X[F], Y[F], label = 'Firms', color = 'red')
    plt.scatter(X[D], Y[D], label = 'Delivery', color = 'blue')
    plt.scatter(X[S], Y[S], label = 'Satellites', color = 'green')
    plt.scatter(X[C], Y[C], label = 'Clients', color = 'brown')
    for i in range(XY.shape[0]):
        x = X[i]
        y = Y[i]
        plt.text(x+0.3, y+0.3, i, fontsize=9)
    for s in S:
        for k in V_i[s]:
            for i in N:
                for j in N:
                    key = (i,j,k)
                    if key in w_opt:
                        if w_opt[key] > 0:
                            x1, x2 = XY[i,0], XY[j,0]
                            y1, y2 = XY[i,1], XY[j,1]
                            plt.plot([x1,x2],[y1,y2],
                                     color = colors[k],
                                     linestyle = 'dashed',
                                     label = 'Vehicle %s (%s)' % (k, dictveh[k]) if i == dictveh[k] else "")
    plt.legend()
    plt.title('Vehicles from Satellite')
    if save:
        plt.tight_layout()
        plt.savefig('%s-sat.png' % filename, dpi = 250)
    plt.show()

def RecoverOriginalqValues(data, DictRoutes):
    DEM = data['DEM']
    V_i = data['V_i']
    S = data['S']
    D = data['D']
    F = data['F']
    P = data['P']
    P_f = data['P_f']
    q_subp = {}
    # vehicles from satellites
    for s in S:
        DEM[s] = np.zeros(len(P), dtype=int)
        for k in V_i[s]:
            if k in DictRoutes.keys():
                if DictRoutes[k]:
                    # Cummulative demand for vehicle
                    sumdemand = np.zeros(len(P), dtype=int)
                    # Last node visited
                    for t in DictRoutes[k][::-1]:
                        i,j = t
                        if j != s:
                            for p in P:
                                sumdemand[p] = sumdemand[p] + DEM[j][p]
                                q_subp[p,i,j,k] = sumdemand[p]
                    DEM[s] = DEM[s] + sumdemand
        DEM[s] = list(DEM[s])
    # vehicles from delivery
    for d in D:
        DEM[d] = np.zeros(len(P), dtype=int)
        for k in V_i[d]:
            if k in DictRoutes.keys():
                if DictRoutes[k]:
                    # Cummulative demand for vehicle
                    sumdemand = np.zeros(len(P), dtype=int)
                    # Last node visited
                    # HERE I NEED TO DELETE THE FREIGHTS FROM PRODUCTS FROM FIRMS WITH VEHICLES
                    for t in DictRoutes[k][::-1]:
                        i,j = t
                        if j != d:
                            if j not in F:
                                for p in P:
                                    sumdemand[p] = sumdemand[p] + DEM[j][p]
                                    q_subp[p,i,j,k] = sumdemand[p]
                            else:
                                PminPf = [p for p in P if p not in P_f[j]]
                                for p in P_f[j]:
                                    q_subp[p,i,j,k] = 0
                                    aux = max([value for key, value in q_subp.items() if key[0] == p and key[3] == k])
                                    sumdemand[p] = sumdemand[p] - aux
                                for p in PminPf:
                                    aux = max([value for key, value in q_subp.items() if key[0] == p and key[3] == k])
                                    q_subp[p,i,j,k] = aux
                                
                    DEM[d] = DEM[d] + sumdemand
        DEM[d] = list(DEM[d])
    # vehicles from firms
#    for f in F:
#        for k in V_i[f]:
#            if k in DictRoutes.keys():
#                if DictRoutes[k]:
#                    # Cummulative demand for vehicle
#                    sumdemand = np.zeros(len(P), dtype=int)
#                    # Last node visited
#                    for t in DictRoutes[k][::-1]:
#                        i,j = t
#                        if j != f:
#                            for p in P:
#                                sumdemand[p] = sumdemand[p] + DEM[j][p]
#                                q_subp[p,i,j,k] = sumdemand[p]
#    for p in P
    return q_subp

def SaveSolHeuristic(data, file, dt, soldir, q_final, w_final, u_final, y_final, DictRoutes, Opt):

    #Create Excell Writter
    writer = pd.ExcelWriter(os.path.join(soldir, file), engine='xlsxwriter')

    # Save solutions: q
    q_final = RecoverOriginalqValues(data, DictRoutes)
    dfq = []
    for key, value in dict(q_final).items():
        if value > 0:
            #                #print(key, value)
            dfq.append([*key, value])
    dfq = pd.DataFrame(data=dfq, columns=['p', 'i', 'j', 'k', 'q_final'])
    dfq.to_excel(writer, index=False, sheet_name = "q")
    # Save solutions: w
    dfw = []
    for key, value in dict(w_final).items():
        if value > 0:
            dfw.append([*key, value])
    dfw = pd.DataFrame(data=dfw, columns=['i', 'j', 'k', 'w_final'])
    dfw.to_excel(writer, index=False, sheet_name="w")
    # Save solutions: u
    dfu = []
    for key, value in dict(u_final).items():
        if value > 0:
            dfu.append([key, value])
    dfu = pd.DataFrame(data=dfu, columns=['s', 'u_final'])
    dfu.to_excel(writer, index=False, sheet_name="u")
    dfy = []
    for key, value in dict(y_final).items():
        if value > 0:
            dfy.append([key, value])
    dfy = pd.DataFrame(data=dfy, columns=['k', 'y_final'])
    dfy.to_excel(writer, index=False, sheet_name="y")

    dfo = pd.DataFrame({"Value": [Opt], "Time": [dt]})
    dfo.to_excel(writer, sheet_name='Optimization')

    writer.save()

def ExecuteMultiEchelonFromData(datadir,file, plotdir = None, soldir = None):
    """
    This is for executing the math heuristic for solving the multi echelon bla bla
    For plotdir and soldir, give a valid directory in case you want to save the results
    If you don't provide a directory for plotdir, the script will not save the plots.
    The same zill happen if you don't provide a directory for soldir
    
    """
    data = ReadData(datadir, file)
    ti = time()
    q_final, w_final, u_final, y_final, DictRoutes, Opt = ExecuteMultiEchelon(data, filename = file.replace('.xlsx','-pre'))

    # Get the real q_final
    q_final = RecoverOriginalqValues(data, DictRoutes)
    tf = time()
    dt = tf - ti
    if plotdir:
        plotfile = os.path.join(plotdir, 'solution heur ' + file.replace('.xlsx',''))
        AuxSubPlot(data, w_final, figsize = (5,5), save = True, filename = plotfile)
    if soldir:
        SaveSolHeuristic(data, file, dt, soldir, q_final, w_final, u_final, y_final, DictRoutes, Opt)

    return Opt, dt


if __name__ == "__main__":
    """EXECUTION"""

    #datadir = os.path.join(os.path.pardir,'data')
    #soldir = os.path.join(os.path.pardir,'solutions')
    #plotdir = os.path.join(os.path.pardir,'plots')
    #sumdir = os.path.join(os.path.pardir,'summaries')
    plotdir = None
    #soldir = None

    datadir = os.path.join(os.path.pardir,'data')
    soldir = os.path.join(os.path.pardir,'solutions')
    #plotdir = os.path.join(os.path.pardir,'plots-v2')
    sumdir = os.path.join(os.path.pardir,'summaries')

    # Execute for 1 instance
    #file = 'v8-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    #Opt, dt = ExecuteMultiEchelonFromData(datadir,file, plotdir, soldir)

    # Execute for a set of instances
    filetype = 'v%s-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    times = []
    opts = []
    n_instances = 1000


    files_ = [filetype % (i+1) for i in range(n_instances)]
    files = []
    for file in files_:
        ##print(file)
        Opt, dt = ExecuteMultiEchelonFromData(datadir, file, plotdir, soldir)
        files.append(file)
        opts.append(Opt)
        times.append(dt)
    df_heur = pd.DataFrame(data = {'Instance' : files,
                                   'Math Heur Obj' : opts,
                                   'Math Heur time (sec)' : times})
    df_heur.to_excel(os.path.join(sumdir, filetype.replace('v%s-', 'summary-heur-v2-%sinstances-' % n_instances)), index = False)
