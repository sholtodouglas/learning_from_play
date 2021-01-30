import os
import numpy as np
import tensorflow as tf
def save_weights(path, actor, configs, dl, run_id, encoder=None, planner=None, step=""):
    os.makedirs(path, exist_ok=True)

    print('Saving model weights...')
    # Save the standardisation params
    np.savez(path + 'hyper_params', relative_obs=dl.relative_obs, relative_act=dl.relative_act,
              quaternion_act=dl.quaternion_act,
              joints=dl.joints, LAYER_SIZE=config['layer_size'], LATENT_DIM=config['latent_dim'],
              GCBC=config['gcbc'], PROBABILISTIC=config['num_distribs'], QUANTISED=config['qbits'], run_id=run_id)
    # save timestepped version
    if step != "":
        actor.save_weights(path + 'model_' + str(step) + '.h5')
        if planner is not None: planner.save_weights(path + 'planner_' + str(step) + '.h5')
        if encoder is not None: encoder.save_weights(path + 'encoder_' + str(step) + '.h5')

    # save the latest version
    actor.save_weights(path + 'model.h5')
    if planner is not None: planner.save_weights(path + 'planner.h5')
    if encoder is not None: encoder.save_weights(path + 'encoder.h5')

    # save the optimizer state
    np.save(os.path.join(path, 'optimizer'), optimizer.get_weights())

def load_weights(path, actor, encoder=None, planner=None, step=""):
    actor.load_weights(f'{path}/model' + step + '.h5')
    if planner is not None: planner.load_weights(f'{path}/planner' + step + '.h5')
    if encoder is not None: encoder.load_weights(f'{path}/encoder' + step + '.h5')

def load_optimizer_state(optimizer, load_path, strategy,trainable_variables):
    def optimizer_step():
        # need to do this to initialize the optimiser

        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in trainable_variables]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in trainable_variables]

        # Apply gradients which don't do anything
        optimizer.apply_gradients(zip(zero_grads, trainable_variables))

        # Reload variables
        [x.assign(y) for x, y in zip(trainable_variables, saved_vars)]
        return 0.0

    @tf.function
    def distributed_opt_step():
        '''
        Only used for optimizer checkpointing - we need to run a pass to initialise all the optimizer weights. Can't use restore as colab TPUs don't have a local filesystem.
        '''
        per_replica_losses = strategy.run(optimizer_step, args=())
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    # Load optimizer weights
    opt_weights = np.load(load_path + 'optimizer.npy', allow_pickle=True)

    # init the optimiser
    distributed_opt_step()
    # Set the weights of the optimizer
    optimizer.set_weights(opt_weights)
