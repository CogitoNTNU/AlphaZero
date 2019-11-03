import Main
from multiprocessing import Process, Queue, Pool, Manager
import numpy as np
import time


def multiprocess_function(num_processes, game, agent, config, num_sim=20, games=1, num_search=400):
    result_queue = Queue()
    res_dict=Manager().dict()
    result=list()


    workers = [Process(target=Main.generate_data,
                       args=(x, Main.Gamelogic.FourInARow(), res_dict, config, games))
               for x in range(num_processes)]
    for worker in workers: worker.start()
    # print("start")
    # print("workers", workers)
    for worker in workers:
        # print("workers1", workers)
        worker.join()
        result.append(worker.exitcode)
        # print("exitcode", worker.exitcode)
    print("done")
    print(result)
    # pool = Pool(num_processes)
    # pool.map(Main.generate_data, [(None, Main.Gamelogic.FourInARow(), res_dict, config) for _ in range(num_processes)])
    # pool.close()
    # pool.join()
    # with Pool(processes=num_processes) as pool:
    #     results = pool.starmap(Main.generate_data, [None, game, res_dict, config])
    # print(results)

    now=time.time()
    # print(res_dict)
    print("size", len(res_dict.keys()))
    # result = [result_queue.get(block=False) for _ in range(num_processes)]
    print(time.time()-now)
    return None
    # return np.concatenate([x[0] for x in result]), np.concatenate([x[1] for x in result]),\
    #        np.concatenate([x[2] for x in result])
