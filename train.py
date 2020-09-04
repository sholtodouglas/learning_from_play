import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class BetaScheduler():
    def __init__(self, schedule='constant', beta=0.0, beta_max=1.0, max_steps=1e4, plot=True):
        self.beta_min = beta
        self.beta_max = beta_max
        self.max_steps = max_steps

        if schedule=='constant':
            self.scheduler = lambda s: tf.ones_like(s)*beta
        elif schedule=='linear':
            self.scheduler = self.linear_schedule
        elif schedule=='quadratic':
            self.scheduler = self.quadratic_schedule
        else:
            raise NotImplementedError()
        if plot: self._plot_schedule()
    
    def linear_schedule(self, step):
        beta = self.beta_min + (step) * (self.beta_max-self.beta_min)/self.max_steps
        return tf.clip_by_value(beta, self.beta_min, self.beta_max, name='beta_linear')

    def quadratic_schedule(self, step):
        ''' y = (b1-b0)/n^2 * x^2 + b0 '''
        beta = self.beta_min + (step)**2 * (self.beta_max-self.beta_min)/self.max_steps**2
        return tf.clip_by_value(beta, self.beta_min, self.beta_max, name='beta_quadratic')

    def cyclic_schedule(self, step):
        raise NotImplementedError()

    def _plot_schedule(self):
        t = np.arange(self.max_steps)
        plt.plot(t, self.scheduler(t))
        plt.xlabel('Steps')
        plt.ylabel('Beta')