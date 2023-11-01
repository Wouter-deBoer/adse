import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt


#############################################################################
#                           Noise class                                     #
#                                                                           #
# Creates Noise objects, which are basically just distributions             #
# Used to easen and more efficiently work with scipy distributions          #
# The mean and std of a Noise object can be reset by calling it.            #
#############################################################################
class Noise:
    def __init__(self, mean=0., std=1., half=False):
        self.init_std = std
        self.std = std
        self.mean = mean
        if not half:
            self.dist = sp.norm(loc=mean, scale=std)
        else:
            self.dist = sp.halfnorm(loc=mean, scale=std)
        self.half = half

    def __call__(self, mean=None, std=None):
        if mean is None:
            self.mean = self.dist.mean()
        if std is None:
            self.std = self.dist.std()
        if self.half:
            self.dist = sp.halfnorm(loc=mean, scale=std)
            self.mean = mean
            self.std = std
        else:
            self.dist = sp.norm(loc=mean, scale=std)
            self.mean = mean
            self.std = std

    def sample(self):
        return self.dist.rvs()

    # plot pdf/cdf/sf of noise
    def plot(self, func='pdf'):
        xdata = np.linspace(self.dist.ppf(0.0001),self.dist.ppf(0.9999), 10000)

        if func == 'cdf':
            plt.plot(xdata, self.dist.cdf(xdata), label='cdf')
        elif func == 'sf':
            plt.plot(xdata, self.dist.sf(xdata), label='sf')
        else:
            plt.plot(xdata, self.dist.pdf(xdata), label='pdf')


if __name__ == "__main__":
    noise = Noise(mean=1, std=0.05)
    noise_mud = Noise(mean=0.75, std=0.2)

    xdata_noise = np.linspace(noise.dist.ppf(0.0001), noise.dist.ppf(0.9999), 10000)
    xdata_mud = np.linspace(noise_mud.dist.ppf(0.0001), noise_mud.dist.ppf(0.9999), 10000)

    plt.rc('text', usetex=True)
    plt.figure(dpi=300)

    plt.plot(xdata_mud, noise_mud.dist.pdf(xdata_mud), label='$f_{m}$', linestyle='dashed')
    plt.plot(xdata_noise, noise.dist.pdf(xdata_noise), label = '$f_{f}$', linestyle = 'dashed')

    pdf_noise = np.array([noise.dist.pdf(data) for data in np.linspace(0.0,1.5,10000)], dtype=np.float64)
    pdf_mud = np.array([noise_mud.dist.pdf(data) for data in np.linspace(0.0,1.5,10000)], dtype=np.float64)

    ratio = pdf_noise/pdf_mud

    plt.plot(np.linspace(0.0, 1.5, 10000), (1 - ratio / (ratio + 1)), label='$P(Z_t|S_t=m)$')
    plt.plot(np.linspace(0.0, 1.5,10000), (ratio/(ratio+1)), label = '$P(Z_t|S_t=f)$')

    plt.title("PDFs and Observation models of two ADS")
    plt.xlabel('Velocity')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper left',fontsize='large')
    #plt.savefig('figure')
    plt.show()
