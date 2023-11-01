from experiment import Experiment
import matplotlib.pyplot as plt
import os
import dill
import numpy as np
from statistics import NormalDist

#####################################################################################
#                               evaluation.py                                       #
#                                                                                   #
# Program to make detailed plots of various data fetched in the experiments.        #
# Uses an experiment file compressed into dill format, for this use data_handler.py #
#                                                                                   #
# Plots learning curves, learned parameters, KLD and OVL                            #
#####################################################################################

def determine_color(scale):
    if scale == 0.5:
        color = 'magenta'
    elif scale == 0.75:
        color = 'red'
    elif scale == 0.9:
        color = 'blue'
    elif scale == 1.0:
        color = 'black'
    elif scale == 1.1:
        color = 'cyan'
    elif scale == 1.25:
        color = 'green'
    elif scale == 1.5:
        color = 'orange'
    return color

def determine_marker(var_mud):
    if var_mud == 0.1:
        marker = '.'
    elif var_mud == 0.2:
        marker = '+'
    elif var_mud == 0.3:
        marker = 'x'
    return marker

def plot_all(data, type = None, only_test = False, show_baseline = True, scenarios = None, scenario_variance = None, method = None, mud_mean = None, mud_variance = None):
    if type is None:
        type = ['scores', 'kld']
    if scenarios is None:
        scenarios = ['free', 'mud', 'equal']
    if scenario_variance is None:
        scenario_variance =  ['high', 'low']
    if method is None:
        method = ['bw', 'fwe']
    if mud_mean is None:
        mud_mean = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    if mud_variance is None:
        mud_variance = [0.1, 0.2, 0.3]

    scores = {'free': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}},
            'mud': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}},
            'equal': {'high': {'bw': (), 'fwe': ()}, 'low': {'bw': (), 'fwe': ()}}
            }

    plt.rc('text', usetex=True)

    for s in data:
        if s in scenarios:
            for v in data[s]:
                if v in scenario_variance:
                    for m in data[s][v]:
                        if m in method:
                            d = data[s][v][m]
                            plt.figure(dpi=100)
                            scale_counter = []
                            max_scores = []
                            max_train_scores = []
                            final_scores = []
                            final_train_scores = []
                            initial_scores = []
                            initial_train_scores = []
                            baselines = []
                            baselines_train = []

                            for i in range(len(d[0])):
                                scale = d[4][i]
                                var_mud = d[5][i]
                                color = determine_color(scale)
                                marker = determine_marker(var_mud)

                                if scale in mud_mean and var_mud in mud_variance:
                                    # test scores
                                    if 'scores' in type:

                                        # make sure that there are no duplicate legend items, for multiple mud variances

                                        if scale not in scale_counter and marker=='+' and marker==5:
                                            plt.plot(range(len(d[0][i])), d[0][i], marker=marker, markersize=12,
                                                     linestyle='dashed', color=color, label=f' $\mu_m$ = {scale}')
                                        else:
                                            plt.plot(range(len(d[0][i])), d[0][i], marker=marker, markersize=12,
                                                     linestyle='dashed', color=color)

                                        #plt.plot(range(len(d[0][i])), d[0][i], marker=marker, markersize=12, linestyle='dashed', color=color, label=f' $\mu_m$ = {scale}')
                                        #plt.plot(range(len(d[0][i])), d[0][i], marker='+', markersize=12, linestyle='dashed', color=color, label=f'Test')
                                        if show_baseline:
                                            #plt.axhline(d[2][i], color=color, linestyle=':')
                                            plt.axhline(d[2][i], color=color, linestyle=':', label = 'Test baseline')
                                        # train scores
                                        if not only_test:
                                            #plt.plot(range(len(d[1][i])), d[1][i], marker='o', markersize=6, linestyle='dashed',color=color, fillstyle='none')
                                            plt.plot(range(len(d[1][i])), d[1][i], marker='o', markersize=6, linestyle='dashed', color='blue', fillstyle='none', label='Training')
                                            if show_baseline:
                                                #plt.axhline(d[3][i],  color=color, linestyle='-.')
                                                plt.axhline(d[3][i], color='blue', linestyle='-.', label='Training baseline')

                                        #plt.suptitle('Scores')
                                        plt.ylabel('MCC')
                                        plt.xlabel('Iterations')

                                        scores[s][v][m] = ([max(d[0][i]), max(d[1][i])],
                                                           [d[0][i][-1], d[1][i][-1]],
                                                           [d[2][i], d[3][i]]
                                                           )

                                        # print(f'{s}, {v}, {m}')
                                        # print(f'max = {max(d[0][i])}, max train = {max(d[1][i])}')
                                        # print(f'final = {d[0][i][-1]}, final train = {d[1][i][-1]}')
                                        # print(f'initial = {d[0][i][0]}, initial train = {d[1][i][0]}')
                                        # print(f'baseline = {d[2][i]}, baseline train = {d[3][i]}')
                                        # print('===========================================================')

                                        max_scores.append(max(d[0][i]))
                                        max_train_scores.append(max(d[1][i]))
                                        final_scores.append(d[0][i][-1])
                                        final_train_scores.append(d[1][i][-1])
                                        initial_scores.append(d[0][i][0])
                                        initial_train_scores.append(d[1][i][0])
                                        baselines.append(d[2][i])
                                        baselines_train.append(d[3][i])

                                    elif 't_model' in type:
                                        #plt.plot(range(len(d[0][i])), d[7][i][0], marker='+', markersize=12, linestyle='dashed', color=color, label='P(m|m) learned')
                                        plt.plot(range(len(d[0][i])), d[7][i][0], marker='+', markersize=12, linestyle='dashed', color='black', label='$P(m|m)$ learned')
                                        #plt.plot(range(len(d[0][i])), d[7][i][1], marker='o', markersize=6, linestyle='dashed', color=color, label='P(m|f) learned', fillstyle='none' )
                                        plt.plot(range(len(d[0][i])), d[7][i][1], marker='o', markersize=6,linestyle='dashed', color='orange', label='$P(m|f)$ learned',fillstyle='none')
                                        plt.axhline(d[10][i][0], color='black', linestyle=':', label='$P(m|m)$')
                                        #plt.axhline(d[10][i][1], color=color, linestyle='-.', label='P(m|f)')
                                        plt.axhline(d[10][i][1], color='orange', linestyle='-.', label='$P(m|f)$')

                                        #plt.suptitle('Convergence: t-model')
                                        plt.ylabel('Probability')
                                        plt.xlabel('Iterations')

                                        print(f'{s}, {v}, {m}')
                                        print(f'Learned P(m|m) = {d[7][i][0][-1]}, Learned P(m|f) = {d[7][i][1][-1]}')
                                        print(f' True P(m|m) = {d[10][i][0]}, True P(m|f) = {d[10][i][1]}')
                                        print(f' Difference P(m|m) = {np.abs(d[7][i][0][-1]-d[10][i][0])}, Difference P(m|f) = {np.abs(d[7][i][1][-1]-d[10][i][1])}')
                                        print('===========================================================')

                                    elif 'noise' in type:
                                        plt.plot(range(len(d[0][i])), d[8][i][0], marker='+', markersize=6, linestyle='dashed', color='magenta', label='$\mu_{f_f}$ learned')
                                        plt.plot(range(len(d[0][i])), d[8][i][1], marker='o', markersize=6, linestyle='dashed', color='magenta', label='$\sigma_{f_f}$ learned', fillstyle='none')
                                        plt.plot(range(len(d[0][i])), d[9][i][0], marker='+', markersize=6, linestyle='dashed', color='green', label='$\mu_{f_m}$ learned')
                                        plt.plot(range(len(d[0][i])), d[9][i][1], marker='o', markersize=6, linestyle='dashed', color='green', label='$\sigma_{f_m}$ learned', fillstyle='none')

                                        plt.axhline(1, color='magenta', linestyle=':', label='$\mu{f_f}$ true')
                                        plt.axhline(0.05, color='magenta', linestyle='-.', label='$\sigma_{f_f}$ true')
                                        plt.axhline(scale, color='green', linestyle=':', label='$\mu_{f_m}$ true')
                                        plt.axhline(0.2, color='green', linestyle='-.', label='$\sigma_{f_m}$ true')

                                        #plt.suptitle('Convergence: Gaussian Noise')
                                        plt.ylabel('Magnitude')
                                        plt.xlabel('Iterations')


                                    elif 'kld' in type:
                                        plt.plot(d[6][i], max(d[0][i]), color=color, marker = '+', label='Test')
                                        #plt.plot(d[6][i], max(d[1][i]), color=color, marker = 'o', fillstyle = 'none', label='Train')
                                        plt.suptitle('KLD vs. MCC')
                                        plt.ylabel('MCC')
                                        plt.xlabel('KLD')

                                    elif 'overlap' in type:
                                        overlap = NormalDist(mu=1.0, sigma=0.05).overlap(NormalDist(mu=d[4][i], sigma=d[5][i]))
                                        plt.plot(overlap, max(d[0][i]), color=color, marker = '+', label='Test')
                                    else:
                                        raise ValueError('Type must be "scores", "t_model", "noise", "kld" or "overlap"')

                                    # for handling duplicate legend
                                    if scale not in scale_counter and marker == '+':
                                        scale_counter.append(scale)
                            print(max_scores)
                            print(f'{s}, {v}, {m}')
                            print(f'max = {np.mean(max_scores)}, max train = {np.mean(max_train_scores)}')
                            print(f'final = {np.mean(final_train_scores)}, final train = {np.mean(final_train_scores)}')
                            print(f'initial = {np.mean(initial_scores)}, initial train = {np.mean(initial_train_scores)}')
                            print(f'baseline = {np.mean(baselines)}, baseline train = {np.mean(baselines_train)}')
                            print('===========================================================')

                            #plt.title(f'Scenario : {s}, Variance = {v}, Learning = {m}')
                            #plt.title(f'Scenario : {s}, Variance = {v}, Learning = {m}, Scale = 0.75')


                            plt.ylim((-0.3, 1.05))

                            # cheating matplotlib legend, for multiple variance plot
                            '''
                            plt.plot(2, -1,color='magenta', label= '$\mu_m$ = 0.5')
                            plt.plot(2, -1, color='red', label='$\mu_m$ = 0.75')
                            plt.plot(2, -1, color='blue', label='$\mu_m$ = 0.9')
                            plt.plot(2, -1, color='black', label='$\mu_m$ = 1.0')
                            plt.plot(2, -1, color='cyan', label='$\mu_m$ = 1.1')
                            plt.plot(2, -1, color='green', label='$\mu_m$ = 1.25')
                            plt.plot(2, -1, color='orange', label='$\mu_m$= 1.5')

                            plt.plot(2,-1,color='grey', linestyle= 'dashed', marker = '.', markersize=12, label = '$\sigma_m$ = 0.1')
                            plt.plot(2, -1, color='grey',  linestyle= 'dashed', marker='+', markersize=12, label='$\sigma_m$ = 0.2')
                            plt.plot(2, -1, color='grey',  linestyle= 'dashed', marker='x', markersize=12, label='$\sigma_m$ = 0.3')
                            '''

                            plt.legend()
                            #plt.savefig(f'../experiments_thesis/figures/{s}_{v}_{m}_allvar_combined_{type}')
                            #plt.savefig(f'../experiments_thesis/figures/{s}_{v}_{m}_0_75_{type}')
    return scores

# calculate KLD of true distribution relative to learned distribution
def get_kld(mean_true, var_true, mean_learned, var_learned):
    return 0.5*(np.log(var_true/var_learned)+(var_learned/var_true)+((mean_learned-mean_true)**2)/var_true - 1)

# sort and filter the experiment data in new dictionary, used to plot KLD and OVL
def sort_kld_ovl(data, scenarios=None, scenario_variance=None, method=None, mud_mean = None, mud_variance = None):
    # flipped low and high, looks nicer than high then low in plot
    kld_data = {'free': {'low': {'bw': [], 'fwe': []}, 'high': {'bw': [], 'fwe': []}},
            'mud': {'low': {'bw': [], 'fwe': []}, 'high': {'bw': [], 'fwe': []}},
            'equal': {'low': {'bw': [], 'fwe': []}, 'high': {'bw': [], 'fwe': []}}
            }
    overlap_data = {}
    if scenarios is None:
        scenarios = ['free', 'mud', 'equal']
    if scenario_variance is None:
        scenario_variance =  ['high', 'low']
    if method is None:
        method = ['bw', 'fwe']
    if mud_mean is None:
        mud_mean = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    if mud_variance is None:
        mud_variance = [0.1, 0.2, 0.3]

    for s in data:
        if s in scenarios:
            for v in data[s]:
                if v in scenario_variance:
                    for m in data[s][v]:
                        if m in method:
                            d = data[s][v][m]
                            for i in range(len(d[0])):
                                if d[4][i] in mud_mean and d[5][i] in mud_variance:

                                    # the first line is for f_m, the second for f_f

                                    #kld = round(get_kld(d[4][i], d[5][i], d[9][i][0][-1], d[9][i][1][-1]), 5)
                                    kld = round(get_kld(1.0, 0.05, d[8][i][0][-1], d[8][i][1][-1]), 5)

                                    kld_data[s][v][m].append((d[4][i], d[5][i], kld))

                                    overlap_full = NormalDist(mu=1.0, sigma=0.05).overlap(NormalDist(mu=d[4][i], sigma=d[5][i]))
                                    overlap = round(overlap_full, 5)
                                    if overlap not in overlap_data:
                                        overlap_data[overlap] = []
                                    overlap_data[overlap].append(max(d[0][i]))

                        else:
                            del kld_data[s][v][m]
                else:
                    del kld_data[s][v]
        else:
            del kld_data[s]

    return kld_data, overlap_data

# first sort with sort_kld_ovl, then use in this function to plot the KLD as barchart
# use with caution, variance labels are hardcoded
def plot_kld_bars(kld_data):
    plt.rc('text', usetex=True)
    variances = ("Low", "High", "Low", "High", "Low", "High")
    # hardcoded order of scenarios
    kld_values = {'bw': [], 'fwe': []}

    x = np.arange(len(variances))
    width = 0.4
    multiplier = 0

    fig, ax = plt.subplots(dpi=300)
    for scenario in kld_data:
        for var in kld_data[scenario]:
            for method in kld_data[scenario][var]:
                kld_values[method].append(np.mean([exp[2] for exp in kld_data[scenario][var][method] if not np.isnan(exp[2])]))

    for method, kld in kld_values.items():
        # ugly but easy for now
        if method == 'bw':
            label = 'BW'
        else:
            label = 'FWE'
        offset = width * multiplier
        rects = ax.bar(x + offset, kld, width, label=label)
        ax.bar_label(rects, padding=2, rotation='vertical')
        multiplier += 1

    ax.set_ylabel('KLD')
    #ax.set_title('KLD of learned $f_f$ relative to actual $f_f$')
    ax.set_xticks(x + 0.5 * width, variances, size=12)
    ax.legend(loc='upper right')

    texts = ['Free', 'Mud', 'Equal']
    for idx, x in enumerate(ax.get_xticks()[::2]):
        #ax.text(x + 0.5, -0.003, texts[idx], size=12, ha='center')
        ax.text(x + 0.5, -0.012, texts[idx], size=12, ha='center')

    # uncomment the 3 lines below and comment f_f for f_m
    #plt.ylim((0, 0.03))
    #ax.text(-0.5, -0.0015, '$\sigma:$')
    #ax.text(-0.8, -0.003, '$\it{Scenario}:$')

    # uncomment the 3 lines below and comment f_m for f_f
    ax.text(-0.5, -0.006, '$\sigma:$')
    ax.text(-0.8, -0.012, '$\it{Scenario}:$')
    plt.ylim((0,0.12))

    fig.subplots_adjust(bottom=0.1)
    plt.show()
    #plt.savefig('kld')

# plot the overlap coefficient
def plot_overlap(overlap_data):
    plt.figure(dpi=300)
    plt.rc('text', usetex=True)
    for idx, overlap in enumerate(overlap_data):
        plt.scatter([overlap]*len(overlap_data[overlap]), overlap_data[overlap], color = 'grey', facecolors = 'none')
        plt.scatter(overlap, np.mean(overlap_data[overlap]), marker= '+', color = 'black', s=100, linewidths=3, label='Mean of cluster' if idx == 0 else '')
    plt.legend()
    plt.ylabel('MCC')
    plt.xlabel('Overlap Coefficient')
    plt.title('Maximum MCC vs OVL for all test data and both learning methods')
    #plt.savefig('overlap')

if __name__ == "__main__":
    # insert file path to file created by data_handler.py
    with open('data', 'rb') as file:
        data = dill.load(file)

    # change to
    scenarios = ['equal']
    variance = ['low']
    method = ['bw']
    scale = [0.75]

    scores = plot_all(data, only_test=True, show_baseline=True, type = 'scores', scenarios = scenarios, scenario_variance = variance, method = method, mud_mean = scale, mud_variance = [0.2])
    kld_data, overlap_data = sort_kld_ovl(data, scenarios = scenarios, scenario_variance = variance, method = method, mud_mean = scale, mud_variance = [0.2])

    #plot_overlap(overlap_data)
    plot_kld_bars(kld_data)



