#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:36:18 2020

@author: cristian

multi-echelon MILP V2

"""

# Libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from gurobipy import *
from matplotlib import cm
from time import perf_counter
from scripts.utils import *

# Auxiliary functions

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

def ReadData(datadir, file):
    ti = perf_counter()
    print("Reading data...")
    # Nodes
    xls = pd.ExcelFile(os.path.join(datadir, file))
    df = pd.read_excel(xls, sheet_name = 'Nodes')
    XY = df[['X','Y']].values
    F = df[df['Type'] == 'F'].index.tolist()
    D = df[df['Type'] == 'D'].index.tolist()
    S = df[df['Type'] == 'S'].index.tolist()
    C = df[df['Type'] == 'C'].index.tolist()
    LEZ = dict(zip(df.index.tolist(), df['LEZ'].tolist()))
    city = dict(zip(df.index.tolist(), df['City'].tolist()))
    # Products
    df = pd.read_excel(xls, sheet_name = 'Products')
    P = df['Product'].tolist()
    nu = df['Volume'].tolist()
    omega = df['Weight'].tolist()
    omegaeff = df['Weight eff'].tolist()
    P_f = {}
    for f in F:
        P_f[f] = df[df['Firm'] == f]['Product'].tolist()
    # Demands
    df = pd.read_excel(xls, sheet_name = 'Demands')
    DEM = {}
    for c in C:
        DEM[c] = df[df['Customer'] == c]['Demand'].tolist()
    # Depots cap.
    df = pd.read_excel(xls, sheet_name = 'Depots cap.')
    Lambd = {}
    Omega = {}
    epsil = {}
    for i in range(df.shape[0]):
        d = int(df['Depot'].iloc[i])
        Lambd[d] = df['Lambd'].iloc[i]
        Omega[d] = df['Omega'].iloc[i]
        epsil[d] = df['epsil'].iloc[i]
    # Vehicles
    df = pd.read_excel(xls, sheet_name = 'Vehicles')
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
                    A[s1, k] = 0
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
    tf = perf_counter()
    print("Read in:", tf - ti)
    return data

def PlotNodes(data, figsize = (20,20)):
    fig, ax = plt.subplots(figsize= figsize)
    XY = data['XY']
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    plt.scatter(XY[F,0],XY[F,1],color='red', label= 'Firms', s = 20)
    plt.scatter(XY[D,0], XY[D,1],color='blue', label = 'Delivering', s = 20)
    plt.scatter(XY[S,0], XY[S,1],color='green', label = 'Satellites', s = 20) 
    plt.scatter(XY[C,0], XY[C,1],color='brown', label = 'Customers', s = 20)
    for i in range(XY.shape[0]):
        ax.annotate(i, (XY[i,0], XY[i,1]))
    plt.show()
    
# Main model
    
def MultiEchelon(data, output_variables = False):
    '''
    Builds model from data dictionary
    output_variables: boolean
    If True it outputs the variables q, w, u, y alongside the model.
    '''

    ti = perf_counter()
    print("Building model...")
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
    LEZ = data['LEZ']
    vehictype = data['vehictype']
    city = data['city']
    A = data['A']
    
    model = Model()
    model.setParam('OutputFlag', 0)
    # Auxiliary sets
    # Some firms don't have vehicles:
    firmswithout = []
    if len(F) > 1:
        # This is for banning vehicles from at least 20%.
        # If the number is too small, just one firm will not have vehicles          
        firmswithout = F[- max(int(len(F)*0.2), 1):]
    N = firmswithout + D + S + C
    NminF = [n for n in N if n not in firmswithout] #we never visit firms with
    FminFw = [f for f in F if f not in firmswithout]
    # vehicles, and we don't care about their associated costs
    NminD = [n for n in N if n not in D]
    NminS = [n for n in N if n not in S]
    NminC = [n for n in N if n not in C]
    FcupD = F + D
    DcupS = D + S
    ScupC = S + C
    Vd = [item for sublist in [V_i[d] for d in D] for item in sublist]
    Vs = [item for sublist in [V_i[s] for s in S] for item in sublist]
    VDminVd = {}
    for d in D:
        VDminVd[d] = [k for k in Vd if k not in V_i[d]]
    VSminVs = {}
    for s in S:
        VSminVs[s] = [k for k in Vs if k not in V_i[s]]
    PminPf = {}
    for f in F:
        PminPf[f] = [p for p in P if p not in P_f[f]]
    P_fw = [item for sublist in [P_f[f] for f in firmswithout] for item in sublist]
    P_minfw = [p for p in P if p not in P_fw]
    KminVi = {}
    for i in D + S:
        KminVi[i] = [k for k in K if k not in V_i[i]]
    PfromFw = [item for sublist in [P_f[f] for f in firmswithout] for item in sublist]
    n = len(N)

    print("Initial work:", perf_counter() - ti)

    # Integer decision variables
    q, m, x, xx = {}, {}, {}, {}
    for p in P:
        for i in N:
            for j in N:
                for k in K:
                    q[p,i,j,k] = model.addVar(vtype = GRB.INTEGER, name = 'q[%s,%s,%s,%s]' % (p,i,j,k))

    for p in P:
        for i in NminF:
            for k in K:
                m[p,i,k] = model.addVar(vtype = GRB.INTEGER, name = 'm[%s,%s,%s]' % (p,i,k))
    for p in P:
        for f in firmswithout:
            for k in K:
                x[p,f,k] = model.addVar(vtype = GRB.INTEGER, name = 'x[%s,%s,%s]' % (p,f,k))
    for f in FminFw:
        for p in P_f[f]:
            for i in DcupS:
                xx[p,f,i] = model.addVar(vtype = GRB.INTEGER, name = 'xx[%s,%s,%s]' % (p,f,i))


    print("Integer variables:", perf_counter() - ti)

    # Binary decision variables
    y, z, w, u = {}, {}, {}, {}
    for k in K:
        y[k] = model.addVar(vtype = GRB.BINARY, name = 'y[%s]' % k)
    for i in N:
        for j in N:
            for k in K:
                w[i,j,k] = model.addVar(vtype = GRB.BINARY, name = 'w[%s,%s,%s]' % (i,j,k))
    for k in K:
        for c in C:
            z[k,c] = model.addVar(vtype = GRB.BINARY, name = 'z[%s,%s]' % (k,c))
    for s in S:
        u[s] = model.addVar(vtype = GRB.BINARY, name = 'u[%s]' % s)

    print("Desition variables:", perf_counter() - ti)

    # Auxiliary variable for objective function
    satcost = model.addVar(vtype = GRB.CONTINUOUS, name = 'satcost')
    vehcost = model.addVar(vtype = GRB.CONTINUOUS, name = 'vehcost')
    arccost = model.addVar(vtype = GRB.CONTINUOUS, name = 'arccost')
    freightcost = model.addVar(vtype = GRB.CONTINUOUS, name = 'freigthcost')

    ob = model.addVar(vtype = GRB.CONTINUOUS, name = 'ob')

    print("Constants:", perf_counter() - ti)

    """ MAIN CONSTRAINTS """
    
    # Demand
    model.addConstrs(
        m[p,c,k] == DEM[c][p]*z[k,c] for p in P  for c in C for k in K
    )
    
    # Clients are served by one and only one vehicle
    model.addConstrs(
        quicksum(z[k,c] for k in K) == 1 for c in C
    )
    model.addConstrs(
        quicksum(w[i,c,k] for i in N) == z[k,c] for k in K for c in C
    )
    model.addConstrs(
        quicksum(w[c,i,k] for i in N) == z[k,c] for k in K for c in C
    )

    print("Main constraints:", perf_counter() - ti)
    
    # Flow conservation
    # Customers
    model.addConstrs(
        quicksum(q[p,i,c,k] for i in N) - quicksum(q[p,c,l,k] for l in N) == DEM[c][p]*z[k,c] for p in P for c in C for k in K
    )
    # Satellites
    model.addConstrs(
        quicksum(q[p,i,s,k] for i in N) - quicksum(q[p,s,l,k] for l in N) == m[p,s,k] for p in P for s in S for k in KminVi[s]
    )
    model.addConstrs(
        quicksum(quicksum(q[p,s,j,k] for j in N) for k in V_i[s]) == quicksum(m[p,s,k] for k in KminVi[s]) for p in P_fw for s in S
    )
    # Delivering
    model.addConstrs(
        quicksum(q[p,i,d,k] for i in N) - quicksum(q[p,d,l,k] for l in N) == m[p,d,k] for p in P for d in D for k in KminVi[d]
    )
    model.addConstrs(
        quicksum(quicksum(q[p,d,j,k] for j in N) for k in V_i[d]) == quicksum(m[p,d,k] for k in K) for p in P_fw for d in D
    )
    model.addConstrs(
        quicksum(quicksum(q[p,d,j,k] for j in N) for k in V_i[d]) == quicksum(m[p,d,k] for k in KminVi[d]) + quicksum(xx[p,f,d] for f in FminFw) for p in P_minfw for d in D
    )
    
    
    # Firms
    model.addConstrs(
        quicksum(q[p,i,f,k] for i in N) - quicksum(q[p,f,l,k] for l in N) == -x[p,f,k] for p in P for f in firmswithout for k in Vd
    )
    model.addConstrs(
        quicksum(x[p,f,k] for k in Vd) == quicksum(DEM[c][p] for c in C) for f in firmswithout for p in P_f[f]
    )
    # Capacity constraints
    # Delivery
    model.addConstrs(
        quicksum(quicksum(m[p,d,k]*nu[p] for p in P) for k in K) <= Lambd[d] for d in D
    )
    model.addConstrs(
        quicksum(quicksum(m[p,d,k]*omega[p] for p in P) for k in K) <= Omega[d] for d in D
    )
    # Satellites
    model.addConstrs(
        quicksum(quicksum(m[p,s,k]*nu[p] for p in P) for k in K) <= Lambd[s]*u[s] for s in S
    )
    model.addConstrs(
        quicksum(quicksum(m[p,s,k]*omega[p] for p in P) for k in K) <= Omega[s]*u[s] for s in S
    )
    # Vehicles
    model.addConstrs(
        quicksum(quicksum(q[p,i,j,k]*nu[p] for p in P) for j in N) <= Phi[k]*y[k] for i in N for k in K
    )
    model.addConstrs(
        quicksum(quicksum(q[p,i,j,k]*omega[p] for p in P) for j in N) <= Theta[k]*y[k] for i in N for k in K
    )
    model.addConstrs(
        quicksum(quicksum(m[p,j,k]*nu[p] for p in P) for j in NminF) <= Phi[k]*y[k] for k in K
    )
    model.addConstrs(
        quicksum(quicksum(m[p,j,k]*omega[p] for p in P) for j in NminF) <= Theta[k]*y[k] for k in K
    )
    # Arc and vehicle utilization
    model.addConstrs(
        y[k] <= u[s] for s in S for k in V_i[s]
    )
    model.addConstrs(
        w[i,j,k] <= y[k] for i in N for j in N for k in K
    )
    model.addConstrs(
        q[p,i,j,k]*nu[p] <= Phi[k]*w[i,j,k]*A[j,k] for p in P for i in N for j in N for k in K
    )
    model.addConstrs(
        q[p,i,j,k]*omega[p] <= Theta[k]*w[i,j,k]*A[j,k] for p in P for i in N for j in N for k in K
    )
    model.addConstrs(
        quicksum(w[i,j,k] for i in N) == quicksum(w[j,l,k] for l in N) for j in N for k in K
    )
    model.addConstrs(
        quicksum(quicksum(w[d,j,k] for j in N) for k in V_i[d]) <= len(V_i[d]) for d in D
    )
    model.addConstrs(
        quicksum(quicksum(w[s,j,k] for j in N) for k in V_i[s]) <= len(V_i[s])*u[s] for s in S
    )

    print("Firms:", perf_counter() - ti)

    """
    CONSTRAINTS FROM ASSUMPTIONS
    """
    # Vehicle and node compatibility
    model.addConstrs(
        quicksum(w[i,j,k] for i in N) <= A[j,k] for j in N for k in K
    )

    # Counterintuitive paths of vehicles from delivering companies are not allowed
    model.addConstrs(
        w[i,f,k] == 0 for i in ScupC for f in firmswithout for k in Vd
    )
    # Vehicles pass at most once per node:
    model.addConstrs(
        quicksum(w[i,j,k] for i in N) <= 1 for j in N for k in K
    )
    model.addConstrs(
        quicksum(w[j,i,k] for i in N) <= 1 for j in N for k in K
    )
    # Experimental. Vehicles need to pass through their depot and end empty
    model.addConstrs(
        quicksum(w[int(j), i, k] for i in N) >= y[k] for j in V_i.keys() for k in V_i[j]
    )
    model.addConstrs(
        quicksum(quicksum(q[p, i, j, k] for p in P) for i in N) == 0 for j in V_i.keys() for k in V_i[j]
    )

    print("Constraints from assumptions:", perf_counter() - ti)

    """
    LOGICAL CONSTRAINTS
    """
    model.addConstrs(
        w[i,i,k] == 0 for i in N for k in K
    )

    model.addConstrs(
        m[p,s,k] == 0 for p in P_minfw for s in S for k in K
    )
    
    # For fixing bug of products belonging to f's with vehicles
    model.addConstrs(
        m[p,d,k] == 0 for p in P for d in D for k in V_i[d]
    )
    print("Logical constraints:", perf_counter() - ti)




    """
    OBJECTIVE FUNCTION
    """
    model.addConstr(
        satcost == quicksum(u[s]*epsil[s] for s in S)
    )
    model.addConstr(
        vehcost == quicksum(delta[k]*y[k] for k in K)
    )
    model.addConstr(
        arccost == quicksum(quicksum(quicksum(r[i,j]*rho[k]*w[i,j,k] for k in K) for j in N) for i in N)
    )
    model.addConstr(
        freightcost == quicksum(quicksum(quicksum(quicksum(q[p,i,j,k]*omegaeff[p]*gamma[k]*r[i,j]for k in K) for j in N) for i in N) for p in P)
    )
    
    # All
    model.addConstr(
        ob == satcost + vehcost + arccost + freightcost
    )


    model.update()
    
    model.__data = q, w, u, y, m, ob
    
    model.setObjective(ob, GRB.MINIMIZE)
    model.update()

    tf = perf_counter()
    print("Built model in:", tf - ti)
    if output_variables:
        return model, q, w, u, y
    else:
        return model


def ExecuteFromInitial(data, sol, time_limit=300):
    '''
    Execute MILP starting from solution described in "sol"
    '''

    # Unpacking data
    F = data['F']
    D = data['D']
    S = data['S']
    C = data['C']
    P = data['P']
    P_f = data['P_f']
    K = data['K']
    V_i = data['V_i']

    model, q, w, u, y = MultiEchelon(data, output_variables = True)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # Auxiliary sets
    # Some firms don't have vehicles:
    firmswithout = []
    if len(F) > 1:
        # This is for banning vehicles from at least 20%.
        # If the number is too small, just one firm will not have vehicles
        firmswithout = F[- max(int(len(F) * 0.2), 1):]
    N = firmswithout + D + S + C
    Vd = [item for sublist in [V_i[d] for d in D] for item in sublist]
    Vs = [item for sublist in [V_i[s] for s in S] for item in sublist]
    VDminVd = {}
    for d in D:
        VDminVd[d] = [k for k in Vd if k not in V_i[d]]
    VSminVs = {}
    for s in S:
        VSminVs[s] = [k for k in Vs if k not in V_i[s]]
    PminPf = {}
    for f in F:
        PminPf[f] = [p for p in P if p not in P_f[f]]
    P_fw = [item for sublist in [P_f[f] for f in firmswithout] for item in sublist]
    KminVi = {}
    for i in D + S:
        KminVi[i] = [k for k in K if k not in V_i[i]]

    for p in P:
        for i in N:
            for j in N:
                for k in K:
                    name = 'q[%s,%s,%s,%s]' % (p, i, j, k)
                    if (p, i, j, k) in sol["q"]:
                        model.setAttr("Start", q[p,i,j,k], sol["q"][p,i,j,k])
                    else:
                        model.setAttr("Start", q[p,i,j,k], 0)

    # Binary decision variables
    for k in K:
        name = 'y[%s]' % k
        model.setAttr("Start", y[k], sol["y"][(k,)])

    for i in N:
        for j in N:
            for k in K:
                name = 'w[%s,%s,%s]' % (i, j, k)
                if (i, j, k) in sol["w"]:
                    model.setAttr("Start", w[i,j,k], sol["w"][i, j, k])
                else:
                    model.setAttr("Start", w[i, j, k], 0)

    for s in S:

        model.setAttr("Start", u[s], sol["u"][s])

    ti = perf_counter()
    q_final, w_final, u_final, y_final, m_final, Opt = ExecuteModel(data, model, time_limit=time_limit)
    dt = perf_counter() - ti

    return q_final, w_final, u_final, y_final, m_final, Opt, dt
    #save_solution(soldir, file, dt, q_final, w_final, u_final, y_final, m_final, Opt)

# Plotting function

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
        try:
            for k in V_i[i]:
                dictveh[k] = i
        except:
            pass
    K = [item for sublist in [V_i[i] for i in D+S] for item in sublist]
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
#     for f in F:
#         for k in V_i[f]:
#             for i in N:
#                 for j in N:
#                     key = (i,j,k)
#                     if key in w_opt:
#                         if w_opt[key] > 0:
#                             x1, x2 = XY[i,0], XY[j,0]
#                             y1, y2 = XY[i,1], XY[j,1]
#                             plt.plot([x1,x2],[y1,y2],
#                                      color = colors[k],
#                                      linestyle = 'dashed',
#                                      label = 'Vehicle %s (%s)' % (k, dictveh[k]) if i == dictveh[k] else "")
#     plt.legend()
#     plt.title('Vehicles from Firms')
#     if save:
#         plt.tight_layout()
#         plt.savefig('%s-firms.png' % filename, dpi = 250)
    
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
    
# Main exe
    
def ExecuteMultiEchelonFromData(datadir,file, plotdir = None, soldir = None):
    """
    This is for executing the MILP for solving the multi echelon bla bla
    For plotdir and soldir, give a valid directory in case you want to save the results
    If you don't provide a directory for plotdir, the script will not save the plots.
    The same will happen if you don't provide a directory for soldir
    
    """
    data = ReadData(datadir, file)
    ti = perf_counter()
    q_final, w_final, u_final, y_final, m_final, Opt = ExecuteMultiEchelon(data)
    dt = perf_counter() - ti
    if plotdir:
        plotfile = os.path.join(plotdir, 'solution milp ' + file.replace('.xlsx',''))
        AuxSubPlot(data, w_final, figsize = (5,5), save = True, filename = plotfile)
    if soldir:

        save_solution(soldir, file, dt, q_final, w_final, u_final, y_final, m_final, Opt)

    return Opt, dt



# Aux exe

def ExecuteModel(data, model, time_limit = 300):
    ti = perf_counter()
    print("Optimizing model...")
    model.relax() #uncomment for relax
    model.setParam("TimeLimit", time_limit)
    model.setParam(GRB.Param.OutputFlag, 1)

    model.optimize()
#     model.printAttr('X')
    q, w, u, y, m, ob = model.__data
    q_final = model.getAttr('x', q)
    w_final = model.getAttr('x', w)
    u_final = model.getAttr('x', u)
    y_final = model.getAttr('x', y)
    m_final = model.getAttr('x', m)
    Opt = np.round(model.objVal, 3)
    tf = perf_counter()
    print("Optimized model in:", tf - ti)
    return q_final, w_final, u_final, y_final, m_final, Opt
    
def ExecuteMultiEchelon(data):
    model = MultiEchelon(data)
    q_final, w_final, u_final, y_final, m_final, Opt = ExecuteModel(data, model)
    return q_final, w_final, u_final, y_final, m_final, Opt

if __name__ == "__main__":
    """EXECUTION"""
    datadir = os.path.join(os.path.pardir, 'data')
    soldir = os.path.join(os.path.pardir, 'solutions')
    #plotdir = os.path.join(os.path.pardir,'plots')
    sumdir = os.path.join(os.path.pardir, 'summaries')
    plotdir = None

    # Execute for 1 instance
    # file = 'v8-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    file = 'v1-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    Opt, dt = ExecuteMultiEchelonFromData(datadir, file, plotdir, soldir)

    # Execute for a set of instances
    # filetype = 'v%s-city-n15-f2-d1-s4-c8-p1-v1.xlsx'
    # times = []
    # opts = []
    # failed = []
    # n_instances = 1000
    # files_ = [filetype % (i+1) for i in range(n_instances)]
    # files = []
    # fails = []
    # for file in files_:
    #     try:
    #         Opt, dt = ExecuteMultiEchelonFromData(datadir,file, plotdir, soldir)
    #         opts.append(Opt)
    #         times.append(dt)
    #         files.append(file)
    #     except:
    #         print('Failed for %s' % file)
    #         fails.append(file)

    # df_milp = pd.DataFrame(data = {'Instance' : files,
    #                                'MILP Obj' : opts,
    #                                'MILP time (sec)' : times})
    # df_milp.to_excel(os.path.join(sumdir, filetype.replace('v%s-', 'summary-milp-%sinstances-' % n_instances)), index = False)