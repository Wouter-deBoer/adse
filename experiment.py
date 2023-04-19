import matplotlib.pyplot as plt
import numpy as np
import dill
import sklearn.metrics
import copy
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef

from adse import ADSE
from dynamics import Dynamics
from noise import Noise
from simulation import Simulation
from learner import Learner


####################################################################################
# Program to run experiments and plot the results. The experiment can also be      #
# saved as a pickle for later access.                                              #
# Allows for easier running of multiple sets of simulations with multiple settings.#
# To run, simply define a dynamics object and two adse to be compared.             #
# Then call run or run_and_plot.                                                   #
####################################################################################


class Experiment:
    def __init__(self, simulation, test_data = None, test_ads = None):
        self.sim = simulation

        # dynamics environment which serves as test data
        self.test_data = test_data
        self.test_ads = self.binarize(test_ads.reshape(test_ads.shape[0], 2, 1).T)

        self.trueads = None
        self.nohmm = None
        self.trained = None

        self.trueads_binary = None
        self.no_hmm_binary = None
        self.trained_binary = None

        self.trained_mcc = None
        self.nohmm_mcc = None

        self.t_model = None

        self.t_model_estimate = None
        self.t_model_estimates = {}

        self.learner = None
        self.observations = None

        # is initialized at -1, the minimum score for when this score is to be used in learning, for future applications
        self.train_scores = [-1]
        self.test_scores = []

        #logging
        self.gaussians_free = [(self.sim.adse_trained.noise.mean, self.sim.adse_trained.noise.std)]
        self.gaussians_mud = [(self.sim.adse_trained.noise_mud.mean, self.sim.adse_trained.noise_mud.std)]

        self.learned_t_models = [adse_trained.t_model[0]]

        self.test_score_baseline = None
        self.train_score_baseline = None

        self.kl_divergence = None

    # save class instance as pickle
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            dill.dump(self, file)

    # load class instance
    # usage: new_obj = Experiment.load(file_name)
    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as file:
            return dill.load(file)

    ##########################                     #########################################################
    ##########################     EXPERIMENTS     #########################################################
    ##########################                     #########################################################

    # small function to clean up code
    def run_and_stack_beliefs(self, i):
        # set all input to 1
        input_arr = self.sim.dynamics.input_array.copy()
        input_arr[:, :] = 1.

        self.sim.run(input_arr)

        # omit first index since we will not consider the initial belief
        # reshape to make 3-D array
        if i == 0:
            self.trueads = self.sim.dynamics.get_ads.copy().reshape(self.sim.dynamics.get_ads.shape[0], 2, 1)
            self.nohmm = self.sim.adse_no_hmm.get_belief.copy()[1:].reshape(self.sim.dynamics.get_ads.shape[0], 2, 1)
            self.trained = self.sim.adse_trained.get_belief.copy()[1:].reshape(self.sim.dynamics.get_ads.shape[0], 2, 1)
            self.observations = self.sim.adse_trained.observations.copy()
        else:
            self.trueads = np.dstack((self.trueads, self.sim.dynamics.get_ads.copy()))
            self.nohmm = np.dstack((self.nohmm, self.sim.adse_no_hmm.get_belief.copy()[1:]))
            self.trained = np.dstack((self.trained, self.sim.adse_trained.get_belief.copy()[1:]))
            self.observations = np.dstack((self.observations, self.sim.adse_trained.observations.copy()))

    # function to evaluate an ADSE on the test set
    def evaluate(self, adse = None):
        if adse is None:
            adse_eval = copy.deepcopy(self.sim.adse_trained)
        else:
            adse_eval = copy.deepcopy(adse)

        adse_eval.reset()

        for v in self.test_data:
            if adse_eval.t_model is None:
                adse_eval.classify_mud(v, self.sim.dynamics.noise, self.sim.dynamics.noise_mud)
            else:
                adse_eval.classify_mud_markov(v, self.sim.dynamics.noise, self.sim.dynamics.noise_mud)

        test = adse_eval.get_belief.copy()[1:].reshape(self.test_data.shape[0], 2, 1)
        test_binary = self.binarize(test.T)
        score = sklearn.metrics.matthews_corrcoef(test_binary.flatten(), self.test_ads.flatten())

        # only append score if within experiment loop
        if adse is None:
            self.test_scores.append(score)
            print(f'Test score: {score}')
        else:
            self.test_score_baseline = score
            print(f'Baseline test score: {score}')

    # function to run a set of experiments and to store the classifications in arrays
    def run_experiment(self, n_experiments):

        for i in range(n_experiments):
            print(f"{'Iteration ' + str(i):_^30}")
            self.sim.dynamics.reset()
            self.sim.adse_no_hmm.reset()
            self.sim.adse_trained.reset()

            self.run_and_stack_beliefs(i)

        self.trueads, self.nohmm, self.trained, self.observations = self.trueads.T, self.nohmm.T, self.trained.T, self.observations.T

        print('...Done!')
        return self.trueads, self.nohmm, self.trained

    # function to run a set of experiments and to store the classifications in arrays
    # includes baum-welch learning, see learner.py
    def run_experiment_baum_welch(self, n_experiments):

        for i in range(n_experiments):
            print(f"{'Iteration ' + str(i):_^30}")
            self.sim.dynamics.reset()
            self.sim.adse_no_hmm.reset()

            if i == 0:
                self.sim.adse_trained.reset()
            else:
                if self.sim.adse_trained.noise is not None:
                    estimate, gaussians = self.learner.run_bw()

                    self.sim.adse_trained.reset(estimate[0], self.learner.smoothed[0])

                    self.sim.adse_trained.noise(mean=gaussians[0], std = gaussians[1])
                    self.sim.adse_trained.noise_mud(mean=gaussians[2], std = gaussians[3])

                    self.gaussians_free.append((gaussians[0], gaussians[1]))
                    self.gaussians_mud.append((gaussians[2], gaussians[3]))
                else:
                    estimate = self.learner.run()[0]
                    self.sim.adse_trained.reset(estimate, self.learner.smoothed[0])
                # reset with t-model estimate and initial belief
                self.learned_t_models.append(self.learner.t_model_estimate[0])

                print(self.learner.t_model_estimate)

                self.trueads_binary, self.trained_binary = self.binarize(self.trueads.T[i-1:i]),  self.binarize(self.trained.T[i-1:i])

                score = sklearn.metrics.matthews_corrcoef(self.trained_binary.flatten(), self.trueads_binary.flatten())
                print(f'Train score: {score}')
                self.train_scores.append(score)

            self.evaluate()

            self.run_and_stack_beliefs(i)

        self.trueads,  self.nohmm, self.trained, self.observations = self.trueads.T, self.nohmm.T, self.trained.T, self.observations.T

        # remove first score
        self.train_scores.pop(0)
        print('...Done!')
        return self.trueads, self.nohmm, self.trained

    ########################################  FORWARD EXTRACTION  #######################################################
    # function to run a set of experiments and to store the classifications in arrays
    # includes forward extraction learning on the adse_learned
    ##################################################################################################################
    def run_experiment_fwe(self, n_experiments):

        for i in range(n_experiments):
            print(f"{'Iteration ' + str(i):_^30}")
            self.sim.dynamics.reset()
            self.sim.adse_no_hmm.reset()

            if i != 0:
                self.trueads_binary, self.trained_binary = self.binarize(self.trueads.T[i-1:i]),  self.binarize(self.trained.T[i-1:i])

                if self.sim.adse_trained.noise is not None:

                    gaussians = self.learner.fwe_gaussian_estimation()
                    self.sim.adse_trained.noise(mean = gaussians[0], std = gaussians[1])
                    self.sim.adse_trained.noise_mud(mean=gaussians[2], std = gaussians[3])

                    self.gaussians_free.append((gaussians[0], gaussians[1]))
                    self.gaussians_mud.append((gaussians[2], gaussians[3]))

                estimate = self.determine_transition(trueads_binary_slice = self.trained_binary)


                self.sim.adse_trained.reset(estimate)

                self.learned_t_models.append(estimate)

                score = sklearn.metrics.matthews_corrcoef(self.trained_binary.flatten(), self.trueads_binary.flatten())
                print(f'score: {score}')
                self.train_scores.append(score)

            else:
                self.sim.adse_trained.reset()

            self.evaluate()
            self.run_and_stack_beliefs(i)

        self.trueads, self.nohmm, self.trained, self.observations = self.trueads.T, self.nohmm.T, self.trained.T, self.observations.T

        # remove first score
        self.train_scores.pop(0)
        print('...Done!')
        return self.trueads, self.nohmm, self.trained

    ##########################                       #########################################################
    ##########################     DATA HANDLING     #########################################################
    ##########################                       #########################################################

    # turn a belief vector in a 1-D vector of 0s and 1s, where a 0 denotes that 'mud' was the highest estimate, and a 1 that 'free' was the highest estimate
    def binarize(self, array):
        array_binary = np.argmax(array, axis=1).reshape((array.shape[0], 1, array.shape[2]))
        return array_binary

    # suggest a transition model depending on occurrence of abstract states (Forward extraction)
    # is also used with the true ads to gain a transition model estimate 'supervised'
    # also accepts any random vector filled with 0s and 1s
    def determine_transition(self, n_experiments=None, trueads_binary_slice = None):
        if n_experiments is not None:
            self.run_experiment(n_experiments)
            self.binarize(self.trueads)

        if trueads_binary_slice is None:
            trueads_binary = self.trueads_binary
        else:
            trueads_binary = trueads_binary_slice

        free = np.count_nonzero(trueads_binary.flatten())

        m_m, f_f, f_m, m_f = 0, 0, 0, 0

        for array in trueads_binary:

            for idx, value in enumerate(array[0]):
                if idx >= 1:
                    #print(f'{idx} val: {value}')
                    if value == 1:
                        if array[0][idx] == array[0][idx - 1]:
                            f_f += 1
                            #print('f_f')
                        else:
                            m_f += 1
                            #print('m_f')
                    else:
                        if array[0][idx] == array[0][idx - 1]:
                            m_m += 1
                            #print('m_m')
                        else:
                            f_m += 1
                            #print('f_m')
        print(f'm_m: {m_m}')
        print(f'f_f: {f_f}')
        print(f'f_m: {f_m}')
        print(f'm_f: {m_f}')

        count = np.array([len(trueads_binary.flatten()) - free, free])
        if m_m == 0:
            if f_m == 0:
                self.t_model_estimate = np.array([0, 0])
            else:
                self.t_model_estimate = np.array([0, f_m/(f_f+f_m)])
        elif f_m == 0:
            self.t_model_estimate = np.array([m_m / (m_m + m_f), 0])
        else:
            self.t_model_estimate = np.array([m_m/(m_m+m_f), f_m/(f_f+f_m)])


        ''''#count patch length
            
        mud_length = []
        free_length = []
        for array in trueads_binary:
            counter = [array[0][0]]
            for value in array[0][1:]:
                if value == counter[-1]:
                    counter.append(value)
                elif value == 1:
                    free_length.append(len(counter))
                    counter = [value]
                elif value == 0:
                    mud_length.append(len(counter))
                    counter = [value]
        print(np.mean(mud_length))
        print(np.mean(free_length))
        '''

        print(f'Out of {len(trueads_binary.flatten())} states, {count[0]} were "mud" and {count[1]} were "free"')
        print(f'Suggested transition_model: {self.t_model_estimate}')

        return self.t_model_estimate


    ##########################                   #########################################################
    ##########################     PLOTTING      #########################################################
    ##########################                   #########################################################

    def plot_confusion_matrix(self, plot_title=None):
        cm_trained = confusion_matrix(self.trained_binary.flatten(), self.trueads_binary.flatten())
        cm_no_hmm = confusion_matrix(self.no_hmm_binary.flatten(), self.trueads_binary.flatten())

        fig = plt.figure(figsize=(12, 4))
        if self.sim.dynamics.randomize_mud:
            if isinstance(self.sim.dynamics.randomizer, (list, tuple, np.ndarray)):
                if isinstance(self.sim.dynamics.randomizer[0], Noise):
                    fig.suptitle(
                        f"Random patches: Free(\u03BC = {self.sim.dynamics.randomizer[0].mean}, \u03C3 = {self.sim.dynamics.randomizer[0].std}); Mud(\u03BC = {self.sim.dynamics.randomizer[1].mean}, \u03C3 = {self.sim.dynamics.randomizer[1].std}); t-model: {self.sim.adse_trained.t_model[0]}",
                        fontsize='medium')
                else:
                    fig.suptitle(
                        f"Random: [p(m|m),p(f|f)] = {self.sim.dynamics.randomizer}; t-model: {self.sim.adse_trained.t_model[0]}",
                        fontsize='medium')
            elif isinstance(self.sim.dynamics.randomizer, Noise):
                fig.suptitle(
                    f"Random patches: \u03BC = {self.sim.dynamics.randomizer.mean}, \u03C3 = {self.sim.dynamics.randomizer.std}; t-model: {self.sim.adse_trained.t_model[0]}",
                    fontsize='medium')
        # hardcoded title to describe stationary scenario (e.g. short, long, mixed...)
        else:
            fig.suptitle(f"Stationary, {plot_title}: t-model: {self.sim.adse_trained.t_model[0]}")

        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.title.set_text('Trained')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.title.set_text('No HMM')

        ConfusionMatrixDisplay(confusion_matrix=cm_trained, display_labels=['mud', 'free']).plot(ax=ax1,
                                                                                             values_format='.5g')
        ConfusionMatrixDisplay(confusion_matrix=cm_no_hmm, display_labels=['mud', 'free']).plot(ax=ax2)

        # Matthews correlation coefficient
        self.trained_mcc = sklearn.metrics.matthews_corrcoef(self.trained_binary.flatten(), self.trueads_binary.flatten())
        self.nohmm_mcc = sklearn.metrics.matthews_corrcoef(self.no_hmm_binary.flatten(), self.trueads_binary.flatten())

        print('Trained')
        print(classification_report(self.trained_binary.flatten(), self.trueads_binary.flatten()))
        print(f'MCC score: {self.trained_mcc}')
        print('No HMM')
        print(classification_report(self.no_hmm_binary.flatten(), self.trueads_binary.flatten()))
        print(f'MCC score: {self.nohmm_mcc}')

        return fig

    # plot MCC scores versus iteration (for learning)
    def plot_scores(self):
        plt.figure()
        plt.scatter(range(len(self.test_scores)), self.test_scores)

        plt.title('Learning progress')
        plt.xlabel('Learning step')
        plt.ylabel('MCC score')

    ##########################                   #########################################################
    ##########################      RUNNING      #########################################################
    ##########################                   #########################################################


    # run experiment and prepare data
    def run(self, n_experiments):
        self.run_experiment(n_experiments)
        self.trueads_binary, self.no_hmm_binary, self.trained_binary = self.binarize(self.trueads), self.binarize(self.nohmm), self.binarize(self.trained)

    # run experiment with learning and prepare data
    def run_fwe(self, n_experiments):
        self.run_experiment_fwe(n_experiments)
        self.trueads_binary, self.no_hmm_binary, self.trained_binary = self.binarize(self.trueads), self.binarize(self.nohmm), self.binarize(self.trained)

    def run_baum_welch(self, n_experiments):
        self.run_experiment_baum_welch(n_experiments)
        self.trueads_binary, self.no_hmm_binary, self.trained_binary = self.binarize(self.trueads), self.binarize(self.nohmm), self.binarize(self.trained)

    # self.run, self.run_baum_welch or self.run_fwe but also plot confusion matrix and scores
    def run_and_plot(self, n_experiments, plot_title=None, file_name=None, learning=False):
        if learning == 'fwe':
            self.run_fwe(n_experiments)
            self.plot_scores()
        elif learning == 'bw':
            self.run_baum_welch(n_experiments)
            self.plot_scores()
        elif learning is None or 'None':
            self.run(n_experiments)
        else:
            raise ValueError('As argument for "learning", pass None for no learning, "bw" for Baum-Welch learning, or "vb" for Viterbi learning')
        self.plot_confusion_matrix(plot_title=plot_title)

        self.train_score_baseline = sklearn.metrics.matthews_corrcoef(self.no_hmm_binary[-1].flatten(), self.trueads_binary[-1].flatten())
        print(f'Baseline train score: {self.train_score_baseline}')
        self.evaluate(self.sim.adse_no_hmm)

        self.determine_transition()
        if file_name is not None:
            self.save(file_name)


###########################################################################################################
# For running experiments, change:                                                                        #
# mud_means: true means of movement in mud state                                                          #
# mud_variances: true variance of movement in mud state                                                   #
# learning: choose 'bw' for Baum-Welch, 'fwe' for Forward Extraction or both                              #
# randomizer: true likelihood of mud appearing , first entry is p(mud|mud), second is p(free|free)        #
# or Noise object to use in Dynamics.create_patches()                                                     #
#                                                                                                         #
# And change within the Dynamics object:                                                                  #
# n: amount of timesteps of one training iteration
# random_seed: random seed to be used, keep the same number to reproduce experiments
# randomize_mud: enable to allow randomized mud (patches). Disable to allow for patches as in constraints #
# file_name: folder to place results in plus name you would like file to start with                       #
#                                                                                                         #
# plot_title: if using non-randomized mud, name the scenario. Otherwise, set to None                      #
#                                                                                                         #
#                                                                                                         #
###########################################################################################################

if __name__ == "__main__":
    mud_means = [0.75, 0.9]#[0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    mud_variances = [0.2]#[0.1, 0.2, 0.3]
    for learning in ['bw','fwe']:
        for mud_mean_scale in mud_means:
            for mud_var in mud_variances:


                # initialize gaussian distributions for movements; initialize only the variances, the mean of movement in mud is scaled automatically in dynamics.py by mud_mean_scale
                noise = Noise(std=0.05)
                noise_mud = Noise(std=mud_var)

                noise_randomizer = [Noise(mean=8.7, std=3.0), Noise(mean=8.7*mud_mean_scale, std=3.0*mud_mean_scale)]

                dynamics = Dynamics(dt=1, n=500, noise=noise, noise_mud=noise_mud, mud_mean_scale=mud_mean_scale, randomize_mud=True,
                                    randomizer=noise_randomizer, verbose=False, random_seed=3)

                # calculate KL divergence for comparison
                dist_mud_mean = dynamics.mud_mean_scale
                dist_mud_std = noise_mud.std
                kl_divergence = 0.5*(np.log(noise.std/dist_mud_std)+(dist_mud_std/noise.std)+((dist_mud_mean-1)**2)/noise.std - 1)

                # initialise baseline
                adse_no_hmm = ADSE(['mud', 'free'], verbose=False)

                # initialise adse with transition model of [0.5,0.5]
                adse_trained = ADSE(['mud', 'free'], t_model=[0.5, 0.5], verbose=False)

                # add initial gaussian distributions to adse_trained to allow learning of observation model
                adse_trained.noise = Noise(mean=1., std=0.05)
                adse_trained.noise_mud = Noise(mean=0.1, std=1.0)

                # create the test dynamics using the train dynamics
                # ensuring that the test data is generated by the same principle as the train dynamics
                test_dynamics = copy.deepcopy(dynamics)
                test_dynamics.n = 1000

                test_dynamics.random_seed = 2

                # reset to use new random seed
                test_dynamics.reset(keep_input=False)
                test_input_arr = test_dynamics.input_array
                test_input_arr[:, :] = 1.
                test_dynamics.integrate(test_input_arr)

                # use this filename format
                # order is not important, as long as the name contains the scenario name, scenario variance and learning method
                # also change data_handler.py if other format is to be used
                file_name = 'experiments/' + 'equal' + '_' + str(learning) + '_' + 'sigma_low' + '_' + str(dynamics.mud_mean_scale) + '_' + str(mud_var)

                # initialize experiment object
                exp = Experiment(Simulation(dynamics, adse_trained, adse_no_hmm), test_dynamics.get_velocity, test_dynamics.get_ads)
                exp.kl_divergence = kl_divergence
                exp.learner = Learner(adse_trained)

                exp.run_and_plot(3,  plot_title=None, file_name=file_name, learning=learning)