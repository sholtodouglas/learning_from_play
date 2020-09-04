import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class BetaScheduler():
    def __init__(self, schedule='constant', beta=0.0, beta_max=1.0, max_steps=1e4,
                    cycles=10, duty_cycle=0.5, plot=True):
        self.beta_min = beta
        self.beta_max = beta_max
        self.max_steps = max_steps
        self.cycles = cycles
        self.duty_cycle = duty_cycle

        if schedule=='constant':
            self.scheduler = lambda s: tf.ones_like(s, dtype=tf.float32)*beta
        elif schedule=='linear':
            self.scheduler = self.linear_schedule
        elif schedule=='quadratic':
            self.scheduler = self.quadratic_schedule
        elif schedule=='cyclic':
            self.scheduler = self.cyclic_schedule
        else:
            raise NotImplementedError()
        if plot: self._plot_schedule()
    
    def linear_schedule(self, step):
        beta = self.beta_min + (step) * (self.beta_max-self.beta_min)/self.max_steps
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_linear')

    def quadratic_schedule(self, step):
        ''' y = (b1-b0)/n^2 * x^2 + b0 '''
        beta = self.beta_min + (step)**2 * (self.beta_max-self.beta_min)/self.max_steps**2
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_quadratic')

    def cyclic_schedule(self, step):
        period = self.max_steps // self.cycles
        step = step % period # map step to cycle
        if step < period * self.duty_cycle:
            # linear regime
            beta = self.beta_min + (step) * (self.beta_max-self.beta_min)/(period*self.duty_cycle)
        else:
            # constant regime
            beta = self.beta_max
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_cyclic')

    def _plot_schedule(self):
        ts = np.arange(self.max_steps, step=100)
        plt.plot(ts, [self.scheduler(t) for t in ts])
        plt.xlabel('Steps')
        plt.ylabel('Beta')