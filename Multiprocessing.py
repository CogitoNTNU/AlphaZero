import Train
from multiprocessing import Process, Manager
import numpy as np
import time
from FourInARow import Config
# from TicTacToe import Config
from collections import defaultdict


class DataStore:
    def __init__(self, max_epochs_stored):
        self.data = {}
        self.max_epochs_stored = max_epochs_stored
        self.counter = 0

    def put_data(self, x, y_pol, y_val):
        self.data[self.counter] = [x, y_pol, y_val]
        self.counter = (self.counter + 1) % self.max_epochs_stored

    def get_data(self):
        x = []
        y_pol = []
        y_val = []

        for data in self.data.values():
            x.extend(data[0])
            y_pol.extend(data[1])
            y_val.extend(data[2])
        return np.array(x), np.array(y_pol), np.array(y_val)


def multiprocess_function(config, num_processes, num_games_each_process, num_search, name_weights, seeds=None):
    res_dict = Manager().dict()
    x = list()
    y_pol = list()
    y_val = list()

    workers = [Process(target=Train.generate_data,
                       args=(res_dict, config, num_games_each_process, num_search, i, name_weights, seeds[i]))
               for i in range(num_processes)]

    for worker in workers:
        worker.daemon = True
        worker.start()
    for worker in workers: worker.join()

    print("done")

    for value in res_dict.values():
        x.extend(value[0])
        y_pol.extend(value[1])
        y_val.extend(value[2])

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
    agent = ResNet.ResNet.build(h, w, d, 128, Config.policy_output_dim, num_res_blocks=10)
    agent.compile(loss=[softmax_cross_entropy_with_logits, 'mean_squared_error'],
                  optimizer=SGD(lr=0.0005, momentum=0.9))
    agent.load_weights(load_name)
    agent.fit(x=x, y=[y_pol, y_val], batch_size=min(128, len(x)), epochs=2, callbacks=[])
    agent.save_weights(store_name)


def combine_equals(x, y_pol, y_val):
    dd = defaultdict(lambda: [0, None, np.zeros(y_pol[0].shape), 0])
    for i in range(len(x)):
        c = dd[str(x[i])]
        c[0] += 1
        c[1] = x[i]
        c[2] += y_pol[i]
        c[3] += y_val[i]
    x = []
    y_pol = []
    y_val = []
    for value in dd.values():
        x.append(value[1])
        y_pol.append(value[2] / value[0])
        y_val.append(value[3] / value[0])
    x = np.array(x)
    y_pol = np.array(y_pol)
    y_val = np.array(y_val)
    return x, y_pol, y_val


def train(config, epochs, num_processes, num_games_each_process, num_search, game_name):
    h, w, d = config.board_dims[1:]

    data_store = DataStore(4)

    # TODO: create process that does this
    # import ResNet as nn

    base_name = "Models/" + str(game_name) + "/"
    # nn.ResNet().build(h, w, d, 128, config.policy_output_dim, num_res_blocks=10).save_weights(base_name + "10_3_0.h5")

    for epoch in range(epochs):
        now = time.time()
        load_weights_name = base_name + "10_3_" + str(epoch) + ".h5"
        seed_max = 1000000000
        seeds = [[np.random.randint(0, seed_max) for _ in range(num_games_each_process)] for _ in
                 range(num_games_each_process)]
        x, y_pol, y_val = multiprocess_function(config, num_processes, num_games_each_process, num_search,
                                                load_weights_name,
                                                seeds=seeds)

        x, y_pol, y_val = combine_equals(x, y_pol, y_val)

        data_store.max_epochs_stored = min(40, 4 + 3 * epochs // 4)
        data_store.put_data(x, y_pol, y_val)
        x, y_pol, y_val = data_store.get_data()
        store_weights_name = base_name + "10_3_" + str(epoch + 1) + ".h5"
        worker = Process(target=train_process, args=(x, y_pol, y_val, load_weights_name, store_weights_name, h, w, d))
        worker.daemon = True
        worker.start()
        worker.join()
        print("Finished epoch", epoch, "time:", time.time() - now)
    return None


if __name__ == '__main__':
    train(Config, 3000, 8, 500, 600, Config.name)
