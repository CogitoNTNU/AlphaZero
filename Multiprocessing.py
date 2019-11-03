import Main
from multiprocessing import Process, Queue, Pool, Manager
import numpy as np
import time


def multiprocess_function(num_processes, game, agent, config, num_sim=20, games=1, num_search=400):
    result_queue = Queue()
    res_dict = Manager().dict()
    x = list()
    y_pol = list()
    y_val = list()

    workers = [Process(target=Main.generate_data,
                       args=(x, Main.Gamelogic.FourInARow(), res_dict, config, games, x))
               for x in range(num_processes)]
    for worker in workers: worker.start()
    for worker in workers: worker.join()

    print("done")
    print(res_dict)

    for value in res_dict.values():
        x.extend(value[0])
        y_pol.extend(value[1])
        y_val.extend(value[2])

    now = time.time()
    print("size", len(res_dict.keys()))
    print(time.time() - now)
    return np.array(x), np.array(y_pol), np.array(y_val)
