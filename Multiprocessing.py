import Main
from multiprocessing import Process, Manager
import numpy as np
import time
from FourInARow import Config
from FourInARow import Gamelogic
# from TicTacToe import Config


def multiprocess_function(config, num_processes, num_games_each_process, num_search, name_weights, seeds=None):
    res_dict = Manager().dict()
    x = list()
    y_pol = list()
    y_val = list()

    workers = [Process(target=Main.generate_data,
                       args=(res_dict, config, num_games_each_process, num_search, i, name_weights, seeds[i]))
               for i in range(num_processes)]

    for worker in workers:
        worker.daemon = True
        worker.start()
    for worker in workers: worker.join()

    print("done")
    # print(res_dict)

    for value in res_dict.values():
        x.extend(value[0])
        y_pol.extend(value[1])
        y_val.extend(value[2])

    # now = time.time()
    # print("size", len(res_dict.keys()))
    # print(time.time() - now)
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


def train(config, epochs, num_processes, num_games_each_process, num_search, game_name):
    h, w, d = config.board_dims[1:]

    # import ResNet as nn

    base_name = "Models/" + str(game_name) + "/"
    # nn.ResNet().build(h, w, d, 128, config.policy_output_dim, num_res_blocks=7).save_weights(base_name + "4r_0.h5")

    for epoch in range(193, epochs):
        now=time.time()
        load_weights_name = base_name + "4r_" + str(epoch) + ".h5"
        seed_max = 1000000000
        seeds = [[np.random.randint(0, seed_max) for _ in range(num_games_each_process)] for _ in
                 range(num_games_each_process)]
        x, y_pol, y_val = multiprocess_function(config, num_processes, num_games_each_process, num_search,
                                                load_weights_name,
                                                seeds=seeds)
        store_weights_name = base_name + "4r_" + str(epoch + 1) + ".h5"
        worker = Process(target=train_process, args=(x, y_pol, y_val, load_weights_name, store_weights_name, h, w, d))
        worker.daemon = True
        worker.start()
        worker.join()
        print("Finished epoch", epoch, "time:", time.time()-now)
        # time.sleep(10)
    return None


if __name__ == '__main__':
    train(Config, 300, 8, 100, 400, Config.name)
