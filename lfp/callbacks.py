from tensorflow.keras.callbacks import Callback

class BetaSchedulerCallback(Callback):
    """ For some reason this doesn't seem to work :/ """
    def __init__(self, schedule):
        super(BetaSchedulerCallback, self).__init__()
        self.schedule = schedule

    def on_train_batch_begin(self, step, logs=None):
        if not hasattr(self.model, "beta"):
            raise ValueError('Optimizer must have a "beta" attribute.')
        # Get the current learning rate from model's optimizer.
        # beta = float(tf.keras.backend.get_value(self.model.beta))
        # Call schedule function to get the scheduled learning rate.
        scheduled_beta = float(self.schedule(step))
        # Set the value back to the optimizer before this epoch starts
        # tf.keras.backend.set_value(self.model.beta, scheduled_beta)
        self.model.beta = scheduled_beta
        print(f"\nStep {step:05}: Beta is {self.model.beta:.1e}.")