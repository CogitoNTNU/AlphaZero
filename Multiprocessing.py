import Main
from multiprocessing import Process, Queue
import numpy as np


def multiprocess_funcion(num_processes, game, agent, config, num_sim=20, games=20, num_search=400):
    result_queue = Queue()

    workers = [Process(target=Main.generate_data,
                       args=(result_queue, game, agent, config))
               for _ in range(num_processes)]
    for worker in workers: worker.start()
    print("start")
    for worker in workers: worker.join()
    result = [result_queue.get() for _ in range(num_processes)]
    return np.concatenate([x[0] for x in result]), np.concatenate([x[1] for x in result]),\
           np.concatenate([x[2] for x in result])
