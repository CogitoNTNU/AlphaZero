import os


# Store game (a list of moves)
def store_list(game_name, history, epoch, outcome, id='', dir='Games'):
    path = 'Data/' + str(game_name) + '/' + dir + '/' + str(epoch)
    with open(path + '/' + str(outcome) + '_' + id + '.txt', 'w') as hist_file:
        for move in history:
            hist_file.write(str(move) + '\n')


# Storing moves from a game and the targets
def store_game(game_name, history, v_targets, p_targets, epoch, outcome, id='', prior_probs=None):
    game_epoch_root = 'Data/' + str(game_name) + '/Games'
    game_epoch_path = game_epoch_root + '/' + str(epoch)
    create_dir_if_not_exist(game_epoch_path, game_epoch_root)

    target_epoch_root = 'Data/' + str(game_name) + '/Targets'
    target_epoch_path = target_epoch_root + '/' + str(epoch)
    create_dir_if_not_exist(target_epoch_path, target_epoch_root)

    game_num = len(os.listdir('Data/' + str(game_name) + '/Games/' + str(epoch)))
    store_list(game_name, history, epoch, outcome, id=str(game_num) + id, dir='Games')
    store_list(game_name, v_targets, epoch, outcome, id=str(game_num) + 'v' + id, dir='Targets')
    store_list(game_name, p_targets, epoch, outcome, id=str(game_num) + 'p' + id, dir='Targets')
    if prior_probs is not None:
        store_list(game_name, prior_probs, epoch, outcome, id=str(game_num) + 'pri' + id, dir='Targets')



# Creating a directory if it does not already exist
def create_dir_if_not_exist(dir_path, parent_dir):
    # Creating directories
    if not str(dir_path.split('/')[-1]) in os.listdir(parent_dir):
        os.mkdir(dir_path)
    else:
        pass
        # print(dir_path, 'already existed')


# Create directories for a new game
def create_directories(game_name):
    root = 'Data/'
    game_dir = root + str(game_name)
    games_path = game_dir + '/Games'
    targets_path = game_dir + '/Targets'
    models_path = game_dir + '/Models'

    directories = [[game_dir, root], [games_path, game_dir], [targets_path, game_dir], [models_path, game_dir]]

    # Creating directories
    for dir_path, parent_dir in directories:
        create_dir_if_not_exist(dir_path, parent_dir)


# Store/load weights for a neural network
def save_model(agent, game_name, epoch_num, id=''):
    agent.save_weights("Data/" + str(game_name) + '/Models/' + str(id) + '_' + str(epoch_num) + '.h5')


def load_model(agent, game_name, epoch_num, id=''):
    agent.load_weights("Data/" + str(game_name) + '/Models/' + str(id) + '_' + str(epoch_num) + '.h5')
