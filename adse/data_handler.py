from experiment import Experiment
import os
import dill
import numpy as np

#####################################################################################
#                               data_handler.py                                     #
#                                                                                   #
# Program to extract data out of an experiment file and to compress in useful size  #
# Pass a folder with saved experiments as argument                                  #
# Change num_experiments in get_data() to the correct number                        #
#####################################################################################
def load_experiments(folder):
    files = [file for file in os.listdir(folder) if not file.endswith('testing') or file.endswith('placeholder')]

    # triple nested dictionary: scenario -> variance -> learning type -> list of experiments
    experiments = {'free': {'high': {'bw': [], 'fwe': []}, 'low': {'bw': [], 'fwe': []}},
                   'mud': {'high': {'bw': [], 'fwe': []}, 'low': {'bw': [], 'fwe': []}},
                   'equal': {'high': {'bw': [], 'fwe': []}, 'low': {'bw': [], 'fwe': []}}
                   }

    scenarios = ['free', 'mud', 'equal']
    for file in files:
        for scenario in scenarios:
            if scenario in file:
                if 'high' in file:
                    if 'bw' in file:
                        experiments[scenario]['high']['bw'].append(Experiment.load(folder + '/' + file))
                    elif 'fwe' in file:
                        experiments[scenario]['high']['fwe'].append(Experiment.load(folder + '/' + file))
                elif 'low' in file:
                    if 'bw' in file:
                        experiments[scenario]['low']['bw'].append(Experiment.load(folder + '/' + file))
                    elif 'fwe' in file:
                        experiments[scenario]['low']['fwe'].append(Experiment.load(folder + '/' + file))
    return experiments

def get_data(experiments):
    data = {'free': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}},
            'mud': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}},
            'equal': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}}
            }
    # change according to available experiments
    scenarios = ['equal'] #['free', 'mud', 'equal']
    var = ['low'] #['low', 'high']
    method = ['bw', 'fwe']

    # amount of means x amount of variances
    num_experiments = 2

    for scenario in scenarios:
        for v in var:
            for m in method:
                data[scenario][v][m] = ([experiments[scenario][v][m][i].test_scores for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].train_scores for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].test_score_baseline for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].train_score_baseline for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].sim.dynamics.mud_mean_scale for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].sim.dynamics.noise_mud.std for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].kl_divergence for i in range(num_experiments)],
                                        [np.array(experiments[scenario][v][m][i].learned_t_models).T for i in range(num_experiments)],
                                        [np.array(experiments[scenario][v][m][i].gaussians_free).T for i in range(num_experiments)],
                                        [np.array(experiments[scenario][v][m][i].gaussians_mud).T for i in range(num_experiments)],
                                        [experiments[scenario][v][m][i].t_model_estimate for i in range(num_experiments)]
                                        )
    return data

def save(file_name, data):
    with open(file_name, 'wb') as file:
        dill.dump(data, file)

if __name__ == "__main__":
    folder = 'experiments'
    experiments = load_experiments(folder)
    data = get_data(experiments)
    del experiments

    save('data', data)
    del data
