import pandas as pd
import os
from scripts.milp import ReadData
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import json
from Formatting import *

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