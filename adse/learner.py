import numpy as np
from adse import ADSE

####################################################################################
#                                 Learner class                                    #
#                                                                                  #
# Adds learning capability to an ADSE class object                                 #
# Contains Baum-Welch learning and Forward Extraction (FWE)                        #
# Pass an ADSE object that went through a simulation run (and thus contains        #
# a velocity array, observation array and belief array) and call run_bw to gain    #
# a transition and observation model estimate                                      #
#                                                                                  #
# the transition model estimate for FWE is done in experiment.py to save           #
# computational time. Future work should place it in this class                    #
####################################################################################

class Learner:
    def __init__(self, adse):
        self.adse = adse

        self.forward = None
        self.backward = None
        self.ksi = None

        # aka 'lambda function' or 'forward-backward probability'
        self.smoothed = None

        self.t_model_estimate = np.zeros_like(self.adse.t_model)

        # for logging
        self.t_model_estimates = None

    def reset(self, adse):
        self.adse = adse
        self.forward = None
        self.backward = None
        self.ksi = None
        self.smoothed = None
        self.t_model_estimate = np.zeros_like(self.adse.t_model)

    # forward pass is just state estimation, which is done by the adse during runtime
    def forward_pass(self):
        self.forward = self.adse.get_belief

    # calculate backward pass { p(z_k+1:t | S_k) } for a time window (omit if whole vector is to be used)
    # window must be given in a tuple with 0-indexing
    # e.g: 12th until 56th timestep : window = (11, 55)

    def backward_pass(self, window=None):

        if window is None:
            observations_slice = self.adse.observations.copy()
        else:
            observations_slice = self.adse.observations.copy()[window[0]:window[1]]

        # initialize first backward message (t+1:t) as 1
        # z_t+1 is an empty sequence so probability of observing is 1
        self.backward = np.zeros((observations_slice.shape[0] + 1, observations_slice.shape[1]))
        self.backward[-1] = 1.

        for k, observation in enumerate(observations_slice[::-1]):
            # start at last index
            k_reversed = len(observations_slice) - 1 - k

            # transpose t_model: transition probabilities are reversed (with regard to forward algorithm)
            for state, slice in enumerate(self.adse.t_model.T):
                # p(z_k+1:t | S_k) = p(z_k+1|S_k+1) * p(z_k+2:t | S_k+1) * p(S_k+1 | S_k)
                self.backward[k_reversed][state] = observation[0] * self.backward[k_reversed + 1][0] * slice[0] + \
                                                   observation[1] * self.backward[k_reversed + 1][1] * slice[1]

            self.backward[k_reversed] /= np.sum(self.backward[k_reversed])

        return self.backward

    # aka forward-backward algorithm
    def smoothing(self):
        self.smoothed = self.forward * self.backward
        for index, _ in enumerate(self.smoothed):
            self.smoothed[index] /= np.sum(self.smoothed[index])

        return self.smoothed

    # estimate the transition model with the Baum-Welch method
    def baum_welch(self):
        # ksi is a three-dimensional array
        # first dimension is timesteps: equal to the amount of transitions
        # then an array of probabilities, same as the transition model
        # the probability of transiting from state i to state j at time t given ALL observations
        # p(m|f) = p(S_k+1 = m, S_k = f)
        # e.g.: [[p(m|m), p(m|f)],
        #       [p(f|m), p(f|f)]]
        # then: ksi(S_k+1, S_k); S_k+1 = f, S_k = m
        # ksi(f, m) = ksi[t][1][0]
        # general: S_k+1 = j, S_k = i

        self.ksi = np.zeros((self.forward.shape[0] - 1, self.forward.shape[1], self.forward.shape[1]))
        #print(f'adse t_model: {self.adse.t_model}')

        for k in range(self.ksi.shape[0]):
            #denominator = self.forward[k] @ self.adse.t_model * self.backward[k+1] @ self.adse.observations[k]
            for j, slice in enumerate(self.adse.t_model):
                for i, t_value in enumerate(slice):
                    # note: observation is one step behind per index, so it is k instead of k+1 (like in the formula)
                    # this is because we have one observation per transition. Normally, each new state instead of a transition emits an observation
                    self.ksi[k][j][i] = self.forward[k][i] * self.backward[k+1][j] * t_value * self.adse.observations[k][j]

            self.ksi[k] /= np.sum(self.ksi[k])

        # sum array in time dimension, results in (2,2) shape array
        # the sum of a column of ksi_sum should equal the same column in smoothed_sum
        # note: omit the last index from smoothed. Take T as the size of ksi (amount of transitions)
        ksi_sum = np.sum(self.ksi, axis = 0)
        smoothed_sum = np.sum(self.smoothed[:-1], axis = 0)

        #print(f'ksi: {ksi_sum}')
        #print(f'smoothed: {smoothed_sum}')

        self.t_model_estimate = ksi_sum/smoothed_sum

        # logging
        if self.t_model_estimates is None:
            self.t_model_estimates = self.t_model_estimate.reshape(1, self.t_model_estimate.shape[0], self.t_model_estimate.shape[1])
        else:
            self.t_model_estimates = np.concatenate((self.t_model_estimates, self.t_model_estimate.reshape(1, self.t_model_estimate.shape[0], self.t_model_estimate.shape[1])))

        return self.t_model_estimate

    # estimate gaussians by Baum-Welch(observation model)
    def gaussian_estimation_bw(self):
        velocities = self.adse.velocities.reshape(self.adse.velocities.shape[0],)

        new_mean_mud = np.sum(self.smoothed.T[0][1:]*velocities)/np.sum(self.smoothed.T[0][1:])
        new_mean_free = np.sum(self.smoothed.T[1][1:]*velocities)/np.sum(self.smoothed.T[1][1:])
        new_std_mud = np.sqrt(np.sum(self.smoothed.T[0][1:]*(velocities-new_mean_mud)**2)/np.sum(self.smoothed.T[0][1:]))
        new_std_free = np.sqrt(np.sum(self.smoothed.T[1][1:]*(velocities-new_mean_free)**2)/np.sum(self.smoothed.T[1][1:]))

        print(f'New free gaussian: N~({new_mean_free}, {new_std_free})')
        print(f'New mud gaussian: N~({new_mean_mud}, {new_std_mud})')

        return new_mean_free, new_std_free, new_mean_mud, new_std_mud

    # estimate observation model by FWE
    def fwe_gaussian_estimation(self):
        self.forward_pass()
        binary = np.zeros_like(self.forward)
        for idx,val in enumerate(self.forward):

            binary[idx][np.argmax(val,axis=0)] = 1

        velocities = self.adse.velocities.reshape(self.adse.velocities.shape[0], )

        new_mean_mud = np.sum(binary.T[0][1:]*velocities)/np.sum(binary.T[0][1:])
        new_mean_free = np.sum(binary.T[1][1:]*velocities)/np.sum(binary.T[1][1:])
        new_std_mud = np.sqrt(np.sum(binary.T[0][1:]*(velocities-new_mean_mud)**2)/np.sum(binary.T[0][1:]))
        new_std_free = np.sqrt(np.sum(binary.T[1][1:] * (velocities - new_mean_free) ** 2) / np.sum(binary.T[1][1:]))

        print(f'New free gaussian: N~({new_mean_free}, {new_std_free})')
        print(f'New mud gaussian: N~({new_mean_mud}, {new_std_mud})')

        return new_mean_free, new_std_free, new_mean_mud, new_std_mud

    # call all functions and return a transition model estimate
    def run(self):
        self.forward_pass()
        self.backward_pass()
        self.smoothing()
        return self.baum_welch()

    #call all functions and insert new noise means and stds in adse
    def run_bw(self):
        self.forward_pass()
        self.backward_pass()
        self.smoothing()
        self.baum_welch()
        return self.t_model_estimate, self.gaussian_estimation_bw()

if __name__ == "__main__":
    pass