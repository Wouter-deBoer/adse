import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from dynamics import Dynamics
from adse import ADSE
from noise import Noise

matplotlib.use('TKAgg')

#############################################################################
#                         Visualization class                               #
#                                                                           #
# Creates a simulation in matplotlib                                        #
# Is used to create the simulation worlds and graphs                        #
# Also updates the images and calls the adse                                #
# Originally was created for an ADSE problem with walls                     #
#                                                                           #
# Future work can include using simulation.py within this class since now   #
# it contains a lot of duplicate functions                                  #
#############################################################################

class Visualization:

    def __init__(self, dynamics_obj=Dynamics()):
        self.dynamics = dynamics_obj
        self.adse = ADSE(['mud', 'free'])
        self.adse_no_hmm = ADSE(['mud', 'free'])

        self.car_y = 0.2
        # car dimensions

        self.car_width, self.car_height = 2., .1

        # bounds for rectangle in animation
        self.bounds = [self.dynamics.constraints_wall[0] - 0.5 * self.car_width,
                       self.dynamics.constraints_wall[1] + 0.5 * self.car_width, self.dynamics.constraints_wall[0],
                       self.dynamics.constraints_wall[1]]
        self.rect = None

        # updatable car images
        self.car = plt.Rectangle((self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height),
                                 self.car_width, self.car_height,
                                 fc='r', lw=2, ec='black', label='Current car location')
        self.car_past = plt.Rectangle((self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height),
                                      self.car_width, self.car_height,
                                      fc='none', lw=2, ec='blue', ls='--', label='Previous car location')

        # collision indicator in plot
        self.collision_marker = None

        # keeps velocities to 1 and -1 when enabled
        self.discrete = True

    # Update the image and dynamics with manual input commands
    # discrete means an input of 1 or -1
    # else any number is accepted
    def update_image_manual(self):
        if self.discrete:
            user_input = input("Type r or l to continue\n")
            correct_input = False
            self.car_past.xy = self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height

            while not correct_input:
                if user_input == "r":
                    self.dynamics.integrate_step(1)

                    self.adse.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)
                    self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)

                    if self.dynamics.collision:
                        self.collision_marker.set_data(self.bounds[1], random.uniform(0.5, 1.0) * self.car_y)

                    correct_input = True

                elif user_input == 'l':
                    self.dynamics.integrate_step(-1)

                    self.adse.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)
                    self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                                  self.dynamics.noise_mud)

                    if self.dynamics.collision:
                        self.collision_marker.set_data(self.bounds[0], random.uniform(0.5, 1.0) * self.car_y)
                    correct_input = True

                else:
                    user_input = input("Try again, type r or l to continue\n")
        else:
            try:
                user_input = float(input("Type any number to continue\n"))
            except ValueError:
                user_input = float(input("Try again, type any NUMBER to continue\n"))

            self.car_past.xy = self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height
            self.dynamics.integrate_step(user_input)

            # call the adse to classify
            self.adse.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                          self.dynamics.noise_mud)
            self.adse_no_hmm.classify_mud(self.dynamics.velocity, self.dynamics.noise,
                                          self.dynamics.noise_mud)

            if self.dynamics.collision:
                if user_input > 0:
                    self.collision_marker.set_data(self.bounds[1], random.uniform(0.5, 1.0) * self.car_y)
                else:
                    self.collision_marker.set_data(self.bounds[0], random.uniform(0.5, 1.0) * self.car_y)

        self.car.xy = self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height

    # Update the image and dynamics by passing an input command
    # used in plot function for stepwise updating (by pressing e.g. enter)
    def update_image_auto(self, input_command):
        self.car_past.xy = self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height

        self.dynamics.integrate_step(input_command)

        self.adse_no_hmm.classify_mud(self.dynamics.velocity,
                                      self.dynamics.noise,
                                      self.dynamics.noise_mud)
        self.adse.classify_mud_markov(self.dynamics.velocity, self.dynamics.noise,
                                      self.dynamics.noise_mud)

        if self.dynamics.collision:
            if input_command > 0:
                self.collision_marker.set_data(self.bounds[1], random.uniform(0.5, 1.0) * self.car_y)
            else:
                self.collision_marker.set_data(self.bounds[0], random.uniform(0.5, 1.0) * self.car_y)

        self.car.xy = self.dynamics.state[0] - 0.5 * self.car_width, self.car_y - self.car_height

    # plot a car simulation, a velocity plot, a sliding window plot and a belief plot
    def plot(self, input_array=None, stepwise=False):
        # enable interactive mode
        plt.ion()

        # use GridSpec to create custom layout
        fig = plt.figure()
        gs = GridSpec(3, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        # initialize rectangle 'world'
        self.rect = plt.Rectangle(self.bounds[::2], self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2],
                                  fc='none', lw=2, ec='black')
        ax1.set_xlim(self.bounds[0] - 1, self.bounds[1] + 1)
        ax1.set_ylim(0, 1)
        ax1.add_patch(self.rect)
        self.collision_marker, = ax1.plot([], [], 'rX', markersize=20, mec='black')

        # initialize velocity plot
        ax2.set_xlim(0, self.dynamics.n)
        ax2.set_ylim(0, 2)
        ax2.set_xlabel('timestep')
        ax2.set_ylabel('velocity')
        line, = ax2.plot([], [], 'ro')
        xdata, ydata = [], []

        # initialize moving window plot
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 2)
        line_window, = ax3.plot([], [], 'ro')

        # initialize mud belief plot
        ax4.set_ylim(0, 1)
        bars_mud = ax4.bar(self.adse.states, self.adse.belief)

        # initialize velocity histogram
        ax5.set_xlim(0, 1.5)
        ax5.set_xlabel('Velocity')
        n_bins = np.linspace(0, 1.5, 30)

        # initialize images
        ax1.add_patch(self.car)
        ax1.add_patch(self.car_past)

        ax1.legend()

        # for a single step
        if input_array is None:
            for i in range(self.dynamics.n):
                self.update_image_manual()

                # if not at a wall, remove the collision marker and re-initialize it
                if not self.dynamics.collision:
                    self.collision_marker.remove()
                    self.collision_marker, = ax1.plot([], [], 'rX', markersize=20, mec='black')
                print(i)

                # adjust data for velocity plots
                xdata.append(i)
                ydata.append(self.dynamics.velocity[0])
                line.set_data(xdata, ydata)

                # adjust sliding window plot
                ax3.set_xlim(i - 10, i)
                line_window.set_data(xdata, ydata)

                # adjust bar plot
                for bar, h in zip(bars_mud, self.adse.belief):
                    bar.set_height(h)

                # adjust histogram
                ax5.cla()
                ax5.set_xlim(0, 1.5)
                ax5.set_xlabel('Velocity')
                ax5.hist(ydata,bins=n_bins,histtype='step',density=True)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        # use stepwise to loop through an array by pressing enter
        # otherwise it loops automatically
        else:
            for i in range(self.dynamics.n):
                if stepwise:
                    input(f"Press enter to continue: timestep {self.dynamics.i}\n")
                self.update_image_auto(input_array[i, :][0])

                # adjust data for velocity plots
                xdata.append(i)
                ydata.append(self.dynamics.velocity[0])
                line.set_data(xdata, ydata)

                # adjust sliding window plot
                ax3.set_xlim(i - 10, i)
                line_window.set_data(xdata, ydata)

                for bar, h in zip(bars_mud, self.adse.belief):
                    bar.set_height(h)

                # adjust histogram
                ax5.cla()
                ax5.set_xlim(0, 1.5)
                ax5.set_xlabel('Velocity')
                ax5.hist(ydata,bins=n_bins,histtype='step',density=True)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            if not stepwise:
                plt.show(block=True)

if __name__ == "__main__":
    noise = Noise(std=0.05)
    noise_mud = Noise(std=0.2)

    dynamics = Dynamics(dt=1, constraints_wall=(0, 100), constraints_mud=((10, 15), (30, 35), (50, 55), (70, 75)),
                        n=100, noise=noise,
                        noise_mud=noise_mud, mud_mean_scale=0.75, randomize_mud=True,
                        randomizer=[Noise(mean=10, std=1), Noise(mean=10, std=1)], verbose=True)

    Env = Visualization(dynamics)

    Env.adse = ADSE(['mud', 'free'], t_model=[0.85171, 0.094570], verbose=False)
    Env.discrete = False

    input_arr = Env.dynamics.input_array
    input_arr[:, :] = 1.

    Env.plot(input_arr, stepwise=True)
