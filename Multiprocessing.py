import Main
from multiprocessing import Process, Queue, Manager
import numpy as np
import time
from FourInARow import Config
from FourInARow import Gamelogic


def multiprocess_function(num_processes, game, agent, config, num_sim=20, games=1, num_search=400):
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


def train_process(x, y_pol, y_val, load_name, store_name, h, w, d):
    # Importing libraries and setting the max gpu usage
    from keras.optimizers import SGD
    from loss import softmax_cross_entropy_with_logits, softmax
    import ResNet
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    set_session(sess)

    # Training the agent and storing the new weights
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=7)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.0005, momentum=0.9))
    agent.load_weights(load_name)
    # print("Epoch")
    # # print("x", x, x.shape)
    # # print("y_pol", y_pol, y_val.shape)
    # # print("y_val", y_val, y_val.shape)
    agent.fit(x=x, y=[y_pol, y_val], batch_size=min(32, len(x)), epochs=2, callbacks=[])
    agent.save_weights(store_name)


def train(game, config, num_filters, num_res_blocks, num_sim=10, epochs=1000000, games_each_epoch=10,
          batch_size=32, num_train_epochs=10):
    h, w, d = config.board_dims[1:]

    for epoch in range(epochs):
        x, y_pol, y_val = multiprocess_function(2, game, None, Config, games=2)
        worker = Process(target=train_process, args=(x, y_pol, y_val, h, w, d))
        worker.start()
        worker.join()
        print("Finished epoch", epoch)
        time.sleep(10)
    return agent


if __name__ == '__main__':
    train(Gamelogic.FourInARow(), Config, 128, 0)
