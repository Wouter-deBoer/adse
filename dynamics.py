import numpy as np
import scipy.stats as sp
from noise import Noise
import copy


#############################################################################
# Describes the dynamics of the world.                                      #
# Also defines the constraints of the world (where walls and mud are)       #
# Makes use of Gaussian noise to model movement                                    #
# For now, the mean of the moved distance is equal to the control input     #
# Use integrate_step for a single integration step                          #
# Use integrate for a sequence(numpy array) of integration steps            #
#############################################################################
class Dynamics:

    def __init__(self, noise=Noise(), noise_mud=Noise(),
                 noise_mud_mean=Noise(half=True), dt=0.1, n=100,
                 initial_state=np.array([0]),
                 constraints_wall=(0, 10000),
                 constraints_mud=np.array(([2, 4], [6, 8])), mud_mean_scale=0.75, randomize_mud=False,
                 randomizer=None, verbose=False, random_seed = None):

        # noise is for regular movement, define std for control input of 1
        # noise_mud is for movement in mud, define std for control input of 1
        # mean_mud is then factor*mean_noise, default factor = 0.75
        # noise_mud_mean is for when the mean of the mud movement depends on another probability distribution
        self.noise = noise
        self.noise_mud = noise_mud
        self.noise_mud_mean = noise_mud_mean

        self.dt = dt
        self.n = n

        # save initial state for reusing and assign to state
        self.initial_state = initial_state
        self.state = initial_state

        self.constraints_wall = np.array(constraints_wall)
        if self.state[0] > self.constraints_wall[1] or self.state[0] < self.constraints_wall[0]:
            raise ValueError(
                f"Initial state value '{self.state[0]}' must be within constraints_wall: ({self.constraints_wall[0]}, {self.constraints_wall[1]})")

        self.mud_mean_scale = mud_mean_scale
        self.randomize_mud = randomize_mud

        self.i = 0
        self.collision = None
        self.in_mud = False

        # logging
        self._state = np.zeros((n + 1, 1))
        self._state[0, :] = initial_state
        self._input = np.zeros((n , 1))
        self._velocity = np.zeros((n , 1))
        self._ads = np.zeros((n, 2))

        # can be used for initializing
        self.input_array = np.zeros((n, 1))

        # if more information needs to be printed(velocity and state)
        self.verbose = verbose

        # use if after every reset the environment + velocities should be the same
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


        ##############################################################################################################
        #                                           RANDOMIZER                                                       #
        # Randomizes the world depending on the type of argument passed ((list,tuple,array),Noise, [Noise_f,Noise_m] #
        #                                                                                                            #
        # when a list, tuple or numpy array is passed as argument for 'randomizer',                                  #
        # randomizer is the likelihood of mud appearing, first entry is p(mud|mud), second is p(free|free)           #
        # automatically implies that p(free|mud) = 1 - p(mud|mud) and p(mud|free) = 1 - p(free|free)                 #
        #                                                                                                            #
        # when a Noise object is passed, the world will be 'filled' with patches with size sampled from that noise   #
        # uses create_patches()                                                                                      #
        #                                                                                                            #
        # when a list of Noise objects is passed(size 2), the first Noise object corresponds to free patches,        #
        # and the second to mud patches                                                                              #
        ##############################################################################################################
        self.randomizer = randomizer
        if randomize_mud:
            if isinstance(randomizer, (list, tuple, np.ndarray)):
                if isinstance(randomizer[0], Noise):
                    self.create_patches()
                elif len(randomizer) != 2:
                    raise ValueError("size of list, tuple or numpy array must be 2 and must contain Noise objects or "
                                     "numbers")
                # else if likelihoods, break

            elif isinstance(randomizer, Noise):
                self.create_patches()
            else:
                raise ValueError(
                    "randomizer must be of type list, tuple or numpy array or (list of 2) Noise object(s)")
        else:
            # comma to make nested array if single patch is passed, keeps checking for mud simpler
            if len(constraints_mud[0]) == 1:
                self.constraints_mud = np.array((constraints_mud,))
            else:
                self.constraints_mud = np.array(constraints_mud)

    # creates patches, always starts with a free patch.
    # if two noise objects are passed, the first object will determine free patches and the second mud patches
    def create_patches(self):
        x = 0
        self.constraints_mud = []
        if np.random.random() <= 0.5:
            start_free = True
        else:
            start_free = False
        if isinstance(self.randomizer, (list, tuple, np.ndarray)):
            # switch the noise objects if the first patch should be mud
            if not start_free:
                randomizer = self.randomizer[::-1]
            else:
                randomizer = self.randomizer
            while x < self.constraints_wall[1]:
                free_patch = [0., 0.]
                mud_patch = [0., 0.]
                for idx, noise_obj in enumerate(randomizer):
                    # make sure first patch is always free (free_patch = mud when start_free = False)
                    # if start_free = False, the first noise object is the mud noise
                    # the second noise object is the free noise
                    sample = noise_obj.sample()
                    # assure that no negative patches exist
                    if sample < 0:
                        sample = 0
                    if idx == 0:
                        free_patch[0] = x
                        free_patch[1] = x + sample
                    else:
                        mud_patch[0] = free_patch[1]
                        mud_patch[1] = mud_patch[0] + sample
                        if start_free:
                            self.constraints_mud.append(mud_patch)
                        else:
                            self.constraints_mud.append(free_patch)
                        x = mud_patch[1]

        else:
            while x < self.constraints_wall[1]:
                sample = self.randomizer.sample()
                # assure that no negative patches exist
                if sample < 0:
                    sample = 0
                patch = [0., 0.]
                patch[0] = x
                patch[1] = x + sample
                self.constraints_mud.append(patch)
                x = patch[1]
            # make sure first patch is always free
            if start_free:
                self.constraints_mud = self.constraints_mud[1::2]
            else:
                self.constraints_mud = self.constraints_mud[::2]
        self.constraints_mud = np.array(self.constraints_mud)

    def reset(self, keep_input = True):
        self.state = self.initial_state.copy()
        self._state = np.zeros((self.n + 1, 1))
        self._state[0, :] = self.initial_state.copy()
        self._input = np.zeros((self.n , 1))
        self._velocity = np.zeros((self.n , 1))
        self._ads = np.zeros((self.n, 2))
        self.i = 0

        # set to False if input has changed
        if not keep_input:
            self.input_array = np.zeros((self.n, 1))

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.randomize_mud:
            if isinstance(self.randomizer, (list, tuple, np.ndarray)):
                if isinstance(self.randomizer[0], Noise):
                    self.create_patches()
            elif isinstance(self.randomizer, Noise):
                self.create_patches()

    # single integration step using Euler forward
    def integrate_step(self, input_command):
        # if mud is in patches use this to check if within mud
        if self.randomizer is None or isinstance(self.randomizer, Noise) or isinstance(self.randomizer[0], Noise):
            # for else loop, if break is not reached else is executed
            for mud_patch in self.constraints_mud:
                if np.logical_and(self.state >= mud_patch[0], self.state < mud_patch[1]).all():
                    self.in_mud = True
                    self._ads[self.i, 0] = 1
                    break
            else:
                self.in_mud = False
                self._ads[self.i, 1] = 1
        # for randomizing mud with randomize factor
        else:
            if not self.in_mud:
                if np.random.random() <= self.randomizer[1]:
                    self.in_mud = False
                    self._ads[self.i, 1] = 1
                else:
                    self.in_mud = True
                    self._ads[self.i, 0] = 1
            else:
                if np.random.random() <= self.randomizer[0]:
                    self.in_mud = True
                    self._ads[self.i, 0] = 1
                else:
                    self.in_mud = False
                    self._ads[self.i, 1] = 1

        # if 0 input, do not add noise
        # if within mud, reduce speed (mu_mud = mud_mean_scale*mu_free) and use different probability distribution
        prev_mean = self.noise.mean

        # scale noise according to mean
        # if zero mean or control input, then use initial std
        if prev_mean != 0 and input_command != 0:
            scale = np.abs(input_command / prev_mean)
            self.noise(mean=input_command, std=self.noise.std * scale)

            # use fixed value of mud mean scale*normal mean
            # std needs no scaling; initial value is linked to mud mean scale
            std_mud = self.noise_mud.std * scale
            self.noise_mud(mean=self.mud_mean_scale * input_command, std=std_mud)

        # to use sample from half normal distribution to create new mean for mud:
        # mud_mean_scale = 1 - self.noise_mud_mean.sample()
        # self.noise_mud(mean=mud_mean_scale*input_command)

        else:
            if input_command == 0:
                self.noise_mud(mean=0, std=self.noise_mud.init_std)
                self.noise(mean=0, std=self.noise.init_std)
            else:
                self.noise_mud(mean=self.mud_mean_scale * input_command,
                               std=np.abs(input_command * self.noise_mud.init_std))
                self.noise(mean=input_command, std=np.abs(input_command * self.noise.init_std))

        if input_command == 0:
            state_dot = 0
        else:
            if self.in_mud:
                state_dot = self.noise_mud.sample()
            else:
                state_dot = self.noise.sample()

        # for now, simple on/off speed
        self.input = input_command

        state_next = self.state + state_dot * self.dt

        # check for collision
        if state_next > self.constraints_wall[1]:
            state_next = np.array([self.constraints_wall[1]])
            self.collision = True
        elif state_next < self.constraints_wall[0]:
            state_next = np.array([self.constraints_wall[0]])
            self.collision = True
        else:
            self.collision = False

        self.velocity = (state_next - self.state) / self.dt

        self.state = state_next

        # logging
        self._state[self.i + 1, :] = self.state
        self._input[self.i , :] = self.input
        self._velocity[self.i , :] = self.velocity

        self.i += 1

        if self.verbose:
            print(f'In mud: {self.in_mud}')
            print(f"Velocity: {self.velocity}\nState: {self.state}")

        return self.state

    # integrate for the whole duration(n)
    # use an int or float for same command every timestep
    # use an array (size should be equal to [n+1, input]) for a sequence of commands
    def integrate(self, input_commands):
        if isinstance(input_commands, (int, float)):
            while self.i < self.n:
                self.integrate_step(input_commands)
        elif isinstance(input_commands, np.ndarray):
            if not np.array_equiv(input_commands, self.input_array):
                raise ValueError(f"Array shape {input_commands.shape} must match shape {self.input_array.shape}")
            for input_command in input_commands:
                self.integrate_step(input_command)
        else:
            raise ValueError(f"Input must be of type int, float or numpy array of shape {self._input.shape}")

        return self.state

    @property
    def get_state(self):
        return self._state

    @property
    def get_input(self):
        return self._input

    @property
    def get_velocity(self):
        return self._velocity

    @property
    def get_ads(self):
        return self._ads


if __name__ == "__main__":
    np.random.seed(1)
    noise = [Noise(mean=150, std=1), Noise(mean=150, std=1)]
    d = Dynamics(dt=1, initial_state=np.array([0]), n = 1000, constraints_wall=(0, 1500),
                 constraints_mud=((10, 15), (30, 35), (50, 55), (70, 75)), verbose=True, randomize_mud=True,
                 randomizer=noise)
    d_copy = copy.deepcopy(d)
    input_arr = d.input_array
    input_arr[:, :] = 1

    d.integrate(input_arr)

    print(d.get_state)
    print(d.get_velocity)
