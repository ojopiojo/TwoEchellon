import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import json

def ReadSolution(datadir, file):
    '''
    Read an .xslx solution file into a solution dictionary
    '''
    # Read data
    variables = ["q", "w", "u", "y", "m"]
    indices = ["p", "i", "j", "k"]
    xls = pd.ExcelFile(os.path.join(datadir, file))
    sol = {}

    for variable in xls.sheet_names:
        # Ignore anything that isn't a variable
        if variable not in variables:
            continue

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

def save_solution(soldir, file, dt, q_final, w_final, u_final, y_final, m_final, Opt):
    '''
    Save a soltion based on output from optimization
    '''

    # Write all the parameters to one sheet
    writer = pd.ExcelWriter(os.path.join(soldir, 'solution milp ' + file), engine='xlsxwriter')

    # Save solutions: q
    dfq = []
    for key, value in dict(q_final).items():
        if value > 0:
            dfq.append([*key, value])

    dfq = pd.DataFrame(data=dfq, columns=['p', 'i', 'j', 'k', 'q_final'])
    dfq.to_excel(writer, sheet_name='q')

    # Save solutions: w
    dfw = []
    for key, value in dict(w_final).items():
        if value > 0:
            dfw.append([*key, value])
    dfw = pd.DataFrame(data=dfw, columns=['i', 'j', 'k', 'w_final'])
    dfw.to_excel(writer, sheet_name='w')

    # Save solutions: u
    dfu = []
    for key, value in dict(u_final).items():
        if value > 0:
            dfu.append([key, value])
    dfu = pd.DataFrame(data=dfu, columns=['s', 'u_final'])
    dfu.to_excel(writer, sheet_name='u')

    # Save solutions: y
    dfy = []
    for key, value in dict(y_final).items():
        if value > 0:
            dfy.append([key, value])
    dfy = pd.DataFrame(data=dfy, columns=['k', 'y_final'])
    dfy.to_excel(writer, sheet_name='y')

    # Save solutions: m
    dfm = []
    for key, value in dict(m_final).items():
        if value > 0:
            dfm.append([*key, value])
    dfm = pd.DataFrame(data=dfm, columns=['p', 'i', 'k', 'm_final'])
    dfm.to_excel(writer, sheet_name='m')

    # Save solutions: OtherData
    dfo = pd.DataFrame({"Value": [Opt], "Time": [dt]})
    dfo.to_excel(writer, sheet_name='Optimization')

    writer.save()