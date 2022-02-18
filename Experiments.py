'''
Define and run the main experiments we'll use
'''

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

    max_load = 0
    for filename in os.listdir(directory):

        sol = Plotting.ReadSolution(soldir, "solution milp " + filename)
        data = ReadData(datadir, filename)
        vehicle_routes, vehicle_tours_x, vehicle_tours_y = Plotting.get_vehicle_routes(data, sol)
        load_use = Plotting.get_load_use(data, sol, vehicle_routes)
        for k, phi in data["Theta"].items():
            if len(load_use[k]) > 0:
                max_load = max(max_load, max(load_use[k]) / phi)

        print(max_load)

    '''
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(filename)
            ti = time.perf_counter()
            Opt, dt = ExecuteMultiEchelonFromData(datadir, filename, plotdir, soldir)
            tf = time.perf_counter()
            print(tf - ti)
    '''