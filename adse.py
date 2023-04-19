import numpy as np
from noise import Noise

class ADSE:
    def __init__(self, states, window=3, t_model=None, noise_init = None, noise_mud_init = None, init_belief=None, verbose=True):

        self.states = states
        self.window = window
        self.belief = np.zeros_like(states, dtype=float)

        for state, belief in enumerate(self.belief):
            self.belief[state] = 1 / self.belief.shape[0]
        self.initial_belief = self.belief.copy()

        # for logging
        self._belief = np.copy(self.belief)
        self.observations = None
        self.velocities = None

        # define t-model as first row: e.g. [p(m|m), p(m|f)]
        # then full transition model is:
        #       [p(m|m)           , p(m|f)           ]
        #       [p(f|m) = 1-p(m|m), p(f|f) = 1-p(m|f)]

        if t_model is not None:
            if len(t_model) == len(states) and all((t <= 1 for t in t_model)):
                if isinstance(t_model, np.ndarray):
                    self.t_model = np.array((t_model, 1 - t_model))
                elif isinstance(t_model, (tuple, list)):
                    t_model = np.array(t_model)
                    self.t_model = np.array((t_model, 1 - t_model))
                else:
                    raise ValueError(f"Input must be of type numpy array, list or tuple")
            else:
                raise ValueError(f'Input must be of length {len(states)} and every entry must be smaller or equal than 1')
        else:
            self.t_model = None

        self.current_ads = None
        self.verbose = verbose

        # initial noise to be updated with EM learning, which will be used as observation model.
        # pass None to disable EM learning
        self.noise = noise_init
        self.noise_mud = noise_mud_init

    # reset ADSE for new iteration
    def reset(self, t_model = None, initial_belief = None):
        if initial_belief is None:
            self.belief = self.initial_belief.copy()
            self._belief = self.initial_belief.copy()
        else:
            self.belief = initial_belief
            self._belief = initial_belief
        self.observations = None
        self.velocities = None
        if t_model is not None:
            if isinstance(t_model, np.ndarray):
                self.t_model = np.array((t_model, 1 - t_model))
            elif isinstance(t_model, (tuple, list)):
                t_model = np.array(t_model)
                self.t_model = np.array((t_model, 1 - t_model))

    # classify mud where the noise distribution of the mud is known
    def classify_mud(self, velocity_data, noise, noise_mud, timestep=0):
        # if list-like, adjust data for specified timestep
        if isinstance(velocity_data, (list, np.ndarray)):
            velocity_data = velocity_data[timestep - len(velocity_data)]

        # if 0 input, belief stays the same (velocity is then automatically 0 as well)
        if velocity_data == 0:
            pass
        else:
            pdf = np.round(noise.dist.pdf(velocity_data), 5)
            pdf_mud = np.round(noise_mud.dist.pdf(velocity_data), 5)

            # protect against divisions by 0
            if pdf == 0:
                self.belief[0] = 1
                self.belief[1] = 0
            elif pdf_mud == 0:
                self.belief[0] = 0
                self.belief[1] = 1
            # normalize to get the belief to sum to 1
            else:
                ratio = pdf / pdf_mud
                normalizer = 1 / (ratio + 1)
                self.belief[1] = ratio * normalizer
                self.belief[0] = 1 - self.belief[1]

        self._belief = np.vstack((self._belief, self.belief))

        if self.verbose:
            print(f"Current belief: {self.states}; {self.belief}\n")

    # classify where mean and variance of mud is known, but the process of entering and exiting mud is a markov process
    # this is basically a hidden markov model with a variational observation model
    # uses the t_model, which is the transition model of size (n_states, n_states), see __init__ for example of mud

    def classify_mud_markov(self, velocity_data, dynamics_noise=None, dynamics_noise_mud=None, timestep=0):
        # if list-like, adjust data for specified timestep
        if isinstance(velocity_data, (list, np.ndarray)):
            velocity_data = velocity_data[timestep - len(velocity_data)]

        # if 0 input, belief stays the same (velocity is then automatically 0 as well)
        # this is a special case, since we assume that transitions only occur with a movement
        if velocity_data == 0:
            if self.observations is None:
                self.observations = np.array([0.5,0.5])
            else:
                self.observations = np.vstack((self.observations, np.array([0.5,0.5])))

        else:
            # propagate previous belief with transition model
            new_belief = np.zeros_like(self.belief)
            for state, slice in enumerate(self.t_model):
                new_belief[state] = slice[0] * self.belief[0] + slice[1] * self.belief[1]

            # use EM learned noise as default. Careful, it does not scale with input data
            if self.noise is None:
                pdf = np.round(dynamics_noise.dist.pdf(velocity_data), 5)
                pdf_mud = np.round(dynamics_noise_mud.dist.pdf(velocity_data), 5)
            else:
                pdf = np.round(self.noise.dist.pdf(velocity_data), 5)
                pdf_mud = np.round(self.noise_mud.dist.pdf(velocity_data), 5)

            # protect against divisions by 0
            if pdf == 0:
                self.belief[0] = 1
                self.belief[1] = 0
                ratio = 0
            elif pdf_mud == 0:
                self.belief[0] = 0
                self.belief[1] = 1
                ratio = 10000
            # update belief with pdf ratio and normalize
            else:
                ratio = pdf / pdf_mud
                if self.verbose:
                    print(f'pdf/pdf_mud: {ratio}')
                new_belief[1] = ratio * new_belief[1]
                self.belief = new_belief / np.sum(new_belief)

            # log observation
            observation = [float(1-(ratio/(ratio+1))), float(ratio/(ratio+1))]

            if self.observations is None:
                self.observations = observation
            else:
                self.observations = np.vstack((self.observations, observation))

            if self.velocities is None:
                self.velocities = velocity_data
            else:
                self.velocities = np.vstack((self.velocities, velocity_data))

        self._belief = np.vstack((self._belief, self.belief))

        if self.verbose:
            print(f"Current belief: {self.states}; {self.belief}\n")

    @property
    def get_belief(self):
        return self._belief
