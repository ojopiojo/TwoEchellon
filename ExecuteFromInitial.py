
from scripts.milp import *
import os
from scripts.milp import ExecuteFromInitial, ReadData
from scripts.heurv import ExecuteMultiEchelon, SaveSolHeuristic
import time
from Plotting import ReadSolution, UchoaToDataNoSOl

if __name__ == "__main__":


    #Define directories and files
    soldir = os.path.join(os.path.curdir, 'ArtificialPoints')
    plotdir = None

    datadir = os.path.join(os.path.curdir, 'ArtificialPoints')
    datafile = "X-n31-k25.vrp"
    read_solfile = "step 1 solution milp X-n31-k25.xlsx"
    write_solfile = "step 2 solution milp X-n31-k25.xlsx"

    # Read data
    data = UchoaToDataNoSOl(datadir, datafile)

    #Execute heuristic optimization
    ti = perf_counter()
    q_final, w_final, u_final, y_final, DictRoutes, Opt = ExecuteMultiEchelon(data)
    dt = perf_counter() - ti

    #Save heuristic solution
    SaveSolHeuristic(data, read_solfile, dt, soldir, q_final, w_final, u_final, y_final, DictRoutes, Opt)

    #Read intermediate solution
    sol = ReadSolution(soldir, read_solfile)

    #Execute from solution
    q_final, w_final, u_final, y_final, m_final, Opt, dt = ExecuteFromInitial(data, sol)
    save_solution(soldir, write_solfile, dt, q_final, w_final, u_final, y_final, m_final, Opt)
