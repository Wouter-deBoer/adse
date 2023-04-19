from dynamics import Dynamics
from adse import ADSE
from noise import Noise
import numpy as np
from learner import Learner


#################################################################################
# Creates a simulation without plotting: more lightweight than visualization.py #
#################################################################################

class Simulation:

    def __init__(self, dynamics_obj, adse_trained, adse_no_hmm):
        self.dynamics = dynamics_obj

        self.adse_trained = adse_trained
        self.adse_no_hmm = adse_no_hmm

        # keeps velocities to 1 and -1 when enabled
        self.discrete = True

        # Update dynamics with manual input commands
        # discrete means an input of 1 or -1
        # else any number is accepted

    def update_manual(self):
        if self.discrete:
            user_input = input("Type r or l to continue\n")
            correct_input = False

            while not correct_input:
                if user_input == "r":
                    self.dynamics.integrate_step(1)

                    self.adse_trained.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                                          self.dynamics.noise_mud)
                    self.adse_no_hmm.classify_mud(self.dynamics.velocity,  self.dynamics.noise,
                                                  self.dynamics.noise_mud)

                    correct_input = True

                elif user_input == 'l':
                    self.dynamics.integrate_step(-1)

                    self.adse_trained.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                                          self.dynamics.noise_mud)
                    self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)

                    correct_input = True
                else:
                    user_input = input("Try again, type r or l to continue\n")
        else:
            try:
                user_input = float(input("Type any number to continue\n"))
            except ValueError:
                user_input = float(input("Try again, type any NUMBER to continue\n"))
            self.dynamics.integrate_step(user_input)

            # call the adse to classify
            self.adse_trained.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)
            self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                          self.dynamics.noise_mud)

    # Update the dynamics by passing an input command
    def update_auto(self, input_command):
        self.dynamics.integrate_step(input_command)

        self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                    self.dynamics.noise_mud)
        self.adse_trained.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                              self.dynamics.noise_mud)

    def run(self, input_array=None, stepwise=False):
        if input_array is None:
            for i in range(self.dynamics.n):
                self.update_manual()
        else:
            for i in range(self.dynamics.n):
                if stepwise:
                    input(f"Press enter to continue: timestep {self.dynamics.i}\n")
                self.update_auto(input_array[i, :][0])


if __name__ == "__main__":
    noise = Noise(std=0.05)
    noise_mud = Noise(std=0.2)

    noise_randomizer = [0.7,0.2]
    noise_randomizer = [Noise(mean=5.2, std=1.), Noise(mean=13.9, std=1.)]

    np.random.seed(1)
    dynamics = Dynamics(dt=1, constraints_wall=(0, 150), constraints_mud=((5, 10), (15, 20)), n=100, noise=noise,
                        noise_mud=noise_mud, mud_mean_scale=0.75, randomize_mud=True, randomizer=noise_randomizer, verbose=True)

    t_model = np.array(([0.5, 0.5]))

    adse_trained = ADSE(['mud', 'free'], t_model=t_model)
    adse_no_hmm = ADSE(['mud', 'free'])
    sim = Simulation(dynamics, adse_trained, adse_no_hmm)

    sim.discrete = False

    input_arr = sim.dynamics.input_array
    input_arr[:, :] = 1.

    sim.run(input_arr)
    l = Learner(adse_trained)
    l.backward_pass()
    l.forward_pass()
    l.smoothing()
