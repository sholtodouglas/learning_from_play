import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar
from tensorflow.distribute import ReduceOp
from tensorflow.keras import mixed_precision
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
import lfp
import os
import wandb
import json

from lfp.metric import record, log_action_breakdown


class BetaScheduler():
    def __init__(self, schedule='constant', beta=0.0, beta_max=1.0, max_steps=1e4,
                 cycles=10, duty_cycle=0.5, plot=True):
        self.schedule = schedule
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




class LFPTrainer():

    def __init__(self, args, actor, dl, encoder=None, planner=None, cnn=None, gripper_cnn=None, img_embed_to_goal_space=None, lang_embed_to_goal_space = None\
                optimizer=Adam, strategy=None, global_batch_size=32):

        self.actor = actor
        self.encoder = encoder
        self.planner = planner
        self.cnn = cnn
        self.gripper_cnn = gripper_cnn
        self.img_embed_to_goal_space = img_embed_to_goal_space
        self.lang_embed_to_goal_space = lang_embed_to_goal_space
        self.strategy = strategy
        self.args = args
        self.dl = dl
        self.global_batch_size = global_batch_size

        if args.fp16:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        grad_clips = ['actor': ]
        if self.args.num_distribs is None: # different sized clips due to different sized losses
            grad_clips = {'actor': 0.06,
                        'encoder': 0.03,
                        'planner': 0.001,
                        'cnn': 10, # TODO find value if doing non de
                        'gripper_cnn': 10.0,
                        'img_embed_to_goal_space': 5.0,
                        'lang_embed_to_goal_space': 5.0,}
        else:
            grad_clips = {'actor': 400.0,
                    'encoder': 30,
                    'planner': 5.0,
                    'cnn': 20, # TODO find value if doing non de
                    'gripper_cnn': 10.0,
                    'img_embed_to_goal_space': 5.0,
                    'lang_embed_to_goal_space': 5.0,}


        self.models = {
            'actor':{'model': actor}
            'encoder':{'model': encoder)
        }
        if not args.discrete:
            self.models['planner'] = {'model':planner}

        if args.images:
            self.models['cnn'] = {'model': cnn}
            self.models['img_embed_to_goal_space'] = {'model': img_embed_to_goal_space}
        if args.gripper_images:
            self.models['gripper_cnn'] = {'model': gripper_cnn}
        if args.lang:
            self.models['lang_embed_to_goal_space'] = {'model': lang_embed_to_goal_space}

        for k, v in models.items():
            models[k]['optimiser'] = optimizer(learning_rate=args.learning_rate, clipnorm = grad_clips[v['model']])


        self.nll_action_loss = lambda y, p_y: tf.reduce_sum(-p_y.log_prob(y), axis=2)
        self.mae_action_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.mse_action_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.metrics = {}
        self.metrics['train_loss'] = tf.keras.metrics.Mean(name='train_loss')
        self.metrics['valid_loss'] = tf.keras.metrics.Mean(name='valid_loss')
        for k,v in models.items():
            self.metrics[f"{k}_grad_norm"] = tf.keras.metrics.Mean(name='f"{k}_grad_norm"')

        self.metrics['global_grad_norm'] = tf.keras.metrics.Mean(name='global_grad_norm')

        self.metrics['train_act_with_enc_loss'] = tf.keras.metrics.Mean(name='train_act_with_enc_loss')
        self.metrics['train_act_with_plan_loss'] = tf.keras.metrics.Mean(name='train_act_with_plan_loss')
        self.metrics['valid_act_with_enc_loss'] = tf.keras.metrics.Mean(name='valid_act_with_enc_loss')
        self.metrics['valid_act_with_plan_loss'] = tf.keras.metrics.Mean(name='valid_act_with_plan_loss')

        self.metrics['train_reg_loss'] = tf.keras.metrics.Mean(name='reg_loss')
        self.metrics['valid_reg_loss'] = tf.keras.metrics.Mean(name='valid_reg_loss')

        self.metrics['valid_position_loss'] = tf.keras.metrics.Mean(name='valid_position_loss')
        self.metrics['valid_max_position_loss'] = lfp.metric.MaxMetric(name='valid_max_position_loss')
        self.metrics['valid_rotation_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')
        self.metrics['valid_max_rotation_loss'] = lfp.metric.MaxMetric(name='valid_max_rotation_loss')
        self.metrics['valid_gripper_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')

        self.metrics['valid_enc_position_loss'] = tf.keras.metrics.Mean(name='valid_enc_position_loss')
        self.metrics['valid_enc_max_position_loss'] = lfp.metric.MaxMetric(name='valid_enc_max_position_loss')
        self.metrics['valid_enc_rotation_loss'] = tf.keras.metrics.Mean(name='valid_enc_rotation_loss')
        self.metrics['valid_enc_max_rotation_loss'] = lfp.metric.MaxMetric(name='valid_enc_max_rotation_loss')
        self.metrics['valid_enc_gripper_loss'] = tf.keras.metrics.Mean(name='valid_enc_rotation_loss')


        self.metrics['valid_lang_position_loss'] = tf.keras.metrics.Mean(name='valid_position_loss')
        self.metrics['valid_lang_max_position_loss'] = lfp.metric.MaxMetric(name='valid_max_position_loss')
        self.metrics['valid_lang_rotation_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')
        self.metrics['valid_lang_max_rotation_loss'] = lfp.metric.MaxMetric(name='valid_max_rotation_loss')
        self.metrics['valid_lang_gripper_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')

        self.chkpt_manager = None

    def compute_loss(self, labels, predictions, mask, seq_lens, weightings=None):
        if self.args.num_distribs is not None:
            per_example_loss = self.nll_action_loss(labels, predictions) * mask
        else:
            per_example_loss = self.mae_action_loss(labels, predictions) * mask

        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)


    def compute_MAE(self, labels, predictions, mask, seq_lens, weightings=None):
        per_example_loss = self.mae_action_loss(labels, predictions) * mask # B,T,D
        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep -> B,D
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def compute_regularisation_loss(self, plan, encoding):
        # Reverse KL(enc|plan): we want planner to map to encoder (weighted by encoder)
        reg_loss = tfd.kl_divergence(encoding, plan)  # + KL(plan, encoding)
        return tf.nn.compute_average_loss(reg_loss, global_batch_size=self.global_batch_size)

    @staticmethod
    def compute_fp16_grads(optimizer, loss, tape, model):
        scaled_loss = optimizer.get_scaled_loss(loss)
        return tape.gradient(scaled_loss, model.trainable_variables)

    def make_sequences_variable_length(self, batch):
        '''
        This is a rather gross tiling/casting/indexing function - but it very effectively vectorises 
        variable sequence lengths over entire batches rather than in the dataloader, which brings us within 80% of precomputed speed
        while retaining full data aug!
        '''
    
        B = batch['obs'].shape[0]
        # Create a variable seq lens tensor
        seq_lens = tf.random.uniform(shape=[B], minval=self.args.window_size_min, 
                                                maxval=self.args.window_size_max, dtype=tf.int32)
        batch['seq_lens'] = tf.cast(seq_lens, tf.float32) #must be a float for later loss per timestep calcs
        # Create a mask which is length variable seq lens 
        mask = tf.cast(tf.sequence_mask(seq_lens, maxlen=self.args.window_size_max), tf.float32) #  B,T mask
        multiply_mask = tf.expand_dims(mask, -1) # B, T, 1 (for broadcasting)

        batch['masks'] = mask
        batch['obs'] *= multiply_mask
        batch['acts'] *= multiply_mask
        # Save goals for later as it depends whether it is imgs or not
        # Get numbers 0>B to concat with the seq lens for gather_nd
        B_range = tf.range(0, B, delta=1, dtype=tf.int32, name='range') # B
        B_indices = tf.stack([B_range, seq_lens], axis = 1) # B,2
        if self.args.images:
            # get the goals corresponding to each of the seq len ends
            batch['goal_imgs'] = tf.gather_nd(batch['imgs'], B_indices) # B, imgheight, imgwidth, 3
            # tile_dims = tf.constant([1, self.args.window_size_max, 1,1,1], tf.int32) 
            # goals = tf.tile(goals, tile_dims) # B, T, imgheight, imgwidth, 3
            imgs_mask = tf.cast(multiply_mask, tf.uint8)[:,:,:, tf.newaxis, tf.newaxis] # Must be 5 dim because the imgs are B, T, H, W, C
            #batch['goal_imgs'] = goals *imgs_mask
            # End goal specific stuff, start img specific stuff
            batch['imgs'] *= imgs_mask # must be cast as int or this will be SLOW as it converts img to float
            batch['proprioceptive_features'] *= multiply_mask

            if self.args.gripper_images:
                batch['gripper_imgs'] *= imgs_mask # must be cast as int or this will be SLOW as it converts img to float
        else:
            
            goals = tf.gather_nd(batch['goals'], B_indices)[:, tf.newaxis,:] # B, 1, achieved_goal_dim
            tile_dims = tf.constant([1, self.args.window_size_max, 1], tf.int32) 
            goals = tf.tile(goals, tile_dims) # B, T, achieved_goal_dim
            batch['goals'] = goals*multiply_mask# B, T, achieved_goal_dim

        return batch


    def step(self, inputs=None, lang_labelled_inputs=None, external_videos=None): # one of these must not be none, but either can be
        '''
        A function which wraps the shared processing between train and test step
        Maximally vectorises everything for speed's sake at the cost of some readability

        inputs is normal inputs without sentence labels : obs, acts, goals, imgs, proprioceptive_features, goal_imgs, mask

        '''
            
        indices = {'unlabelled': 0, 'labelled':0, 'vids':0}

        states, actions,  goals, imgs, proprioceptive_features, goal_imgs, masks, sentence_embeddings = None, None, None, None, None, None, None, None

        if inputs is not None:
            states, actions,  goals, seq_lens, masks = inputs['obs'], inputs['acts'], inputs['goals'], inputs['seq_lens'], inputs['masks']
            indices['unlabelled'] = len(actions)
        # We want to concatenate the normal batch, lang_labelled_batch, and external videos, so that we can pass them all through 
        # the encoder at once, at the end, we'd like to return the langlabelled encodings(i.e encodings of lang labelled inputs and 
        # external videos), paired with their appropriate labels so that we can use contrastive losses on them 

        # Get the lengths of each, store that in a dictionary of indices 
         
        # if images, concat imgs, prorpioceptive_features, goal_imgs, not gripper_imgs - that can only be used where we have action labels

        # 1. When using imagesChange the definition of obs_dim to feature encoder dim + proprioceptive features
        # 2. Reshape imgs to B*T H W C.
        # 3. Sub in for states and goals.
        # 4. Then there should be no further changes!
        if self.args.images:
            
            if inputs is not None:
            # [B_unlab,T,H,W,C], [B_unlab,T,D], [B_unlab,H,W,C]
                imgs, proprioceptive_features, goal_imgs, = inputs['imgs'], inputs['proprioceptive_features'], inputs['goal_imgs']
            
            # Frankly only care about these in the context of images
            if self.args.use_language and lang_labelled_inputs is not None:

                indices['labelled'] = indices['unlabelled'] + len(lang_labelled_inputs['acts'])
                if inputs is None:
                    imgs, proprioceptive_features, goal_imgs, actions, masks, seq_lens = lang_labelled_inputs['imgs'], lang_labelled_inputs['proprioceptive_features'], lang_labelled_inputs['goal_imgs'], lang_labelled_inputs['acts'], lang_labelled_inputs['masks'], tf.cast(lang_labelled_inputs['seq_lens'], tf.float32)
                else:
                #  [B_unlab + Blab, T, H, W, C],  [B_unlab + Blab, T, D], [B_unlab + Blab, H, W, C], [B_unlab + Blab, T, D]
                    imgs, proprioceptive_features, goal_imgs, actions, masks, seq_lens = tf.concat([imgs, lang_labelled_inputs['imgs']], 0),\
                                                                tf.concat([proprioceptive_features, lang_labelled_inputs['proprioceptive_features']], 0), \
                                                                tf.concat([goal_imgs, lang_labelled_inputs['goal_imgs']], 0), \
                                                                tf.concat([actions, lang_labelled_inputs['acts']], 0),\
                                                                tf.concat([masks, lang_labelled_inputs['masks']], 0),\
                                                                tf.concat([seq_lens, tf.cast(lang_labelled_inputs['seq_lens'], tf.float32)],0)

                sentence_embeddings = lang_labelled_inputs['label_embeddings']

                
                # contrastive only makes sense in the language context
                if self.args.use_contrastive and external_videos is not None:
                    indices['vids'] =  indices['labelled'] + len(external_videos['imgs'])
                     # B_i +Bll, T, H, W, C
                    imgs, goal_imgs, masks = tf.concat([imgs, external_videos['imgs']], 0), tf.concat([goal_imgs, external_videos['goal_imgs']], 0), tf.concat([masks, external_videos['masks']], 0)  # don't need seq lens from these as it is only used on action level loss
                    sentence_embeddings = tf.concat([sentence_embeddings, external_videos['label_embeddings']], 0)

                # project the sentence embeddings to the same space as the goal
                # B_lab, D
                goal_sentence_embeddings = self.lang_embed_to_goal_space(sentence_embeddings)


            B, T, H, W, C = imgs.shape
            imgs = tf.reshape(imgs, [B * T, H, W, C])
            img_embeddings = tf.reshape(self.cnn(imgs)[0], [B, T, -1]) # [B,T,D]
            states = tf.concat([img_embeddings, proprioceptive_features], -1)  # gets both the image and it's own xyz ori and angle as pose

            goal_img_embeddings = self.cnn(goal_imgs)[0] # [B,D]
            img_in_goal_space = self.img_embed_to_goal_space(goal_img_embeddings) # B, D

            # At this point, we have B_unlab+B_lab+B_vid goal image embeddings, and B_lab+B_vid sentence goal embeddings - all in the goal space!
            # Some fraction of our sentence labelled data should use the image data
            # as we don't have amy sentence labels here, just take all img embeddings
            unlabelled_goal_embeddings = img_in_goal_space[:indices['unlabelled']]
            # we want some images, some sentence embeddings
            if self.args.use_language and lang_labelled_inputs is not None:
                # 0 ..[unlabelled data].......... unlablled ...[image fraction of lang labelled data].... image_fraction .......[lang labelled data].......labelled .................[video data].....................vids
                image_fraction = int(indices['unlabelled'] + ((indices['labelled']-indices['unlabelled']) * self.args.sub_out_language_percent))
                lang_use_img_embeddings = img_in_goal_space[indices['unlabelled']:image_fraction]
                lang_use_lang_embeddings = goal_sentence_embeddings[len(lang_use_img_embeddings):indices['labelled']]
                labelled_goal_embeddings = tf.concat([lang_use_img_embeddings, lang_use_lang_embeddings],0)
                goals = tf.concat([unlabelled_goal_embeddings, labelled_goal_embeddings],0) # Bunlab + Blab, D
                # Same for the vids
                if self.args.use_contrastive:
                  image_fraction = int(indices['unlabelled'] + ((indices['vids']-indices['unlabelled']) * self.args.sub_out_language_percent))
                  vids_use_img_embeddings = img_in_goal_space[indices['labelled']:image_fraction]
                  vids_use_lang_embeddings = goal_sentence_embeddings[len(labelled_goal_embeddings):]
                  video_goal_embeddings = tf.concat([vids_use_img_embeddings, vids_use_lang_embeddings],0)
                  goals = tf.concat([goals, video_goal_embeddings],0)
                # B,1,embedsize
                
            else:
                goals = unlabelled_goal_embeddings
            # The above two will just be 0 if there is nothing from those batches

            # B, T, D
            goals = tf.tile(goals[:,tf.newaxis,:], [1, self.args.window_size_max, 1])
            goals = goals * masks[:, :, tf.newaxis] # B, T, 1 (for broadcasting)

            if self.args.gripper_images:
                if inputs is not None:
                    gripper_imgs = inputs['gripper_imgs']
                if self.args.use_language and lang_labelled_inputs is not None:
                    if inputs is None: # Get rid of this if else hell later TODO
                        gripper_imgs = lang_labelled_inputs['gripper_imgs']
                    else:
                        gripper_imgs = tf.concat([gripper_imgs, lang_labelled_inputs['gripper_imgs']], 0)

                B_grip, _, H_g, W_g, C = gripper_imgs.shape
                gripper_imgs = tf.reshape(gripper_imgs, [B*T, H_g, W_g, C])
                gripper_embeddings = tf.reshape(self.gripper_cnn(gripper_imgs)[0], [B, T, -1]) # should be [B, T, args.gripper_img_embedding_size]
                states = tf.concat([states, gripper_embeddings], -1)



        if self.args.gcbc:
            distrib = self.actor([states, goals])
            return distrib
        else:
            if self.args.encode_all:
                to_encode = tf.concat([states, actions],-1)
            else:
                to_encode = img_embeddings
            
            
            plan = self.planner([states[:, 0, :], goals[:, 0, :]])  # the final goals are tiled out over the entire non masked sequence, so the first timestep is the final goal.
            if self.args.discrete:
                # We want to chunk up the inputs, so each seq goes from B, LEN, EMBED to 
                # that way the lstm encodes little chunks of the sequence
                B,T,D = to_encode.shape
                to_encode = tf.reshape(to_encode, [B*self.args.vq_tiles, T//self.args.vq_tiles, D])#  [B*TILES, LEN/TILES, EMBEDDING]
                
                encoding = self.encoder([to_encode]) # [B*TILES, LATENT] 
                encoding = tf.reshape(encoding[:, :, tf.newaxis], [B, self.args.vq_tiles, -1]) # B, N_TILES, LATENT - so that each tile goes through the gumbel
                
                z_q = tfpl.DistributionLambda(
                        lambda logits: tfd.RelaxedOneHotCategorical(self.args.temperature, encoding)
                    )(encoding)
                z_hard = tf.math.argmax(encoding, axis=-1)  # [B, N_TILES, LATENT]
                
                z_hard = tf.one_hot(z_hard, encoding.shape[-1], dtype=z_q.dtype)  # [B, N_TILES, LATENT]

                z_enc = z_q + tf.stop_gradient(z_hard - z_q)
                z_enc = tf.reshape(z_enc, [B, self.args.vq_tiles, -1]) # Get it back down to batch, Tiles*Encoding where each _ is a tile but in one concatted vector now
                print(z_enc.shape)
            else:
                encoding = self.encoder([to_encode])
                z_enc = encoding.sample()
                z_plan = plan.sample()
                z_enc_tiled = tf.tile(tf.expand_dims(z_enc, 1), (1, self.dl.window_size, 1))
                z_plan_tiled = tf.tile(tf.expand_dims(z_plan, 1), (1, self.dl.window_size, 1))

            enc_policy = self.actor([states, z_enc_tiled, goals])
            if self.args.discrete:
                plan_policy = enc_policy # TODO Concurrently train autoregressive prior
            else:
                plan_policy = self.actor([states, z_plan_tiled, goals])
            return enc_policy, plan_policy, encoding, plan, indices, actions, masks, seq_lens, sentence_embeddings
            # How is step used?
            # In the train and test functions below, where we use enc policy for logloss, plan policy for validation mae, encoding and plan for reg loss
                # Additionally, for contrastive loss we need to get the encodings of the lang labelled and vids only, and their sentence embeddings
                # Then maybe randomly offset by one - layout vid lab vid, or lab vid lab? Then take positive pairs/negative pairs via distance. Ez!
            # Plot just needs the first four...? Meh it can do that itself.
            # In the event that we did contrastive, we get out encodings which will be [b_lab + b_lab + b_vid, D] and the plans which will chase them. 
            # We'll also get indices, unlab, lab, vid
            # therefore, to do contrastive all we need to do is chop off the first B_unlab worth, then the encodings and the sentrence embeddings will be the same length


    def train_step(self, **kwargs):
        inputs, beta, lang_labelled_inputs, external_videos, bulk = kwargs['batch'], kwargs['beta'], kwargs['lang'],kwargs['video'],kwargs['bulk']

        if self.args.bulk_split > 0:
            inputs = {k: tf.concat([inputs[k], bulk[k]], axis=0) for k in inputs.keys()} # combine them

        inputs = self.make_sequences_variable_length(inputs) 

        with tf.GradientTape() as actor_tape, tf.GradientTape() as encoder_tape, tf.GradientTape() as planner_tape, tf.GradientTape() as cnn_tape, tf.GradientTape() as gripper_cnn_tape,\
                                tf.GradientTape() as img_goal_embed_tape, tf.GradientTape() as lang_goal_embed_tape, tf.GradientTape as discrete_projection_tape:

            tapes = [actor_tape encoder_tape, planner_tape, cnn_tape, gripper_cnn_tape, img_goal_embed_tape, lang_goal_embed_tape, discrete_projection_tape]
            tape_idx = 0 # as we use tapes, progress to the next one


            if self.args.gcbc:
                policy = self.step(inputs)
                loss = self.compute_loss(actions, policy, mask, seq_lens)
                gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            else:
                enc_policy, plan_policy, encoding, plan, indices, actions, mask, seq_lens, sentence_embeddings = self.step(inputs, lang_labelled_inputs, external_videos)
                
                act_enc_loss = record(self.compute_loss(actions, enc_policy, mask, seq_lens), self.metrics['train_act_with_enc_loss'])
                if self.args.discrete:
                    loss = act_enc_loss
                else:
                    act_plan_loss = record(self.compute_loss(actions, plan_policy, mask, seq_lens), self.metrics['train_act_with_plan_loss'])
                    reg_loss = record(self.compute_regularisation_loss(plan, encoding), self.metrics['train_reg_loss'])
                    loss = act_enc_loss + reg_loss * beta

                if self.args.fp16:
                    raise NotImplementedError
                else:
                    for k,v in self.models.items():
                        self.models[k]['gradients'] = self.tapes[tape_idx].gradient(loss, v['model'].trainable_variables)
                        tape_idx += 1
                        self.models[k]['norm'] = record(tf.linalg.global_norm(model[k]['gradients']), self.metrics[f'{k}_grad_norm'])
                        self.models[k]['optimizer'].apply_gradients(zip(self.models[k]['gradients'], self.models[k]['model'].trainable_variables))

                
                record(tf.linalg.global_norm(sum[v['gradients'] for k,v in self.models.items()]]), self.metrics['global_grad_norm'])


        return record(loss, self.metrics['train_loss'])


    def test_step(self, **kwargs):
        inputs, beta, lang_labelled_inputs, external_videos = kwargs['batch'], kwargs['beta'], kwargs['lang'], kwargs['video']

        inputs = self.make_sequences_variable_length(inputs) # 
        actions, seq_lens, mask = inputs['acts'], inputs['seq_lens'], inputs['masks']

        if self.args.gcbc:
            policy = self.step(inputs)
            loss = self.compute_loss(actions, policy, mask, seq_lens)
            log_action_breakdown(policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.valid_position_loss, self.valid_max_position_loss, \
                                 self.valid_rotation_loss, self.valid_max_rotation_loss, self.valid_gripper_loss, self.compute_MAE)
        else:
            enc_policy, plan_policy, encoding, plan, indices, actions, mask, seq_lens, sentence_embeddings = self.step(inputs, lang_labelled_inputs, external_videos)
            act_enc_loss = record(self.compute_loss(actions, enc_policy, mask, seq_lens), self.metrics['valid_act_with_enc_loss'])
            
            if self.args.discrete:
                loss = act_enc_loss
                log_action_breakdown(enc_policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.metrics['valid_position_loss'], \
                                 self.metrics['valid_max_position_loss'], self.metrics['valid_rotation_loss'], self.metrics['valid_max_rotation_loss'], self.metrics['valid_gripper_loss'], self.compute_MAE)
            else:
                act_plan_loss = record(self.compute_loss(actions, plan_policy, mask, seq_lens), self.metrics['valid_act_with_plan_loss'])
                reg_loss = record(self.compute_regularisation_loss(plan, encoding), self.metrics['valid_reg_loss'])
                loss = act_plan_loss + reg_loss * beta
                log_action_breakdown(plan_policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.metrics['valid_position_loss'], \
                                 self.metrics['valid_max_position_loss'], self.metrics['valid_rotation_loss'], self.metrics['valid_max_rotation_loss'], self.metrics['valid_gripper_loss'], self.compute_MAE)
                log_action_breakdown(enc_policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.metrics['valid_enc_position_loss'], \
                                 self.metrics['valid_enc_max_position_loss'], self.metrics['valid_enc_rotation_loss'], self.metrics['valid_enc_max_rotation_loss'], self.metrics['valid_enc_gripper_loss'], self.compute_MAE)
                
                if self.args.use_language:
                  # setting probabilistic = false and just passing in the .sample() of the distrib as for some reason slicing it auto samples?
                  log_action_breakdown(plan_policy.sample()[indices['unlabelled']:], actions[indices['unlabelled']:], mask[indices['unlabelled']:], seq_lens[indices['unlabelled']:], False, self.dl.quaternion_act,
                    self.metrics['valid_lang_position_loss'], self.metrics['valid_lang_max_position_loss'], self.metrics['valid_lang_rotation_loss'], self.metrics['valid_lang_max_rotation_loss'], \
                        self.metrics['valid_lang_gripper_loss'], self.compute_MAE)

        return record(loss,self.metrics['valid_loss'])


    @tf.function
    def distributed_train_step(self,inputs):
        per_replica_losses = self.strategy.run(self.train_step, kwargs=inputs)
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


    @tf.function
    def distributed_test_step(self, inputs):
        per_replica_losses = self.strategy.run(self.test_step, kwargs=inputs)
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


    def get_saved_objects(self):
        saved_objects = {'actor': self.actor,
            'encoder': self.encoder,
            'planner': self.planner,
            'actor_optimizer': self.actor_optimizer,
            'encoder_optimizer': self.encoder_optimizer,
            'planner_optimizer': self.planner_optimizer}
        if self.args.images: saved_objects['cnn'], saved_objects['cnn_optimizer'], saved_objects['img_goal_embed'], saved_objects['img_goal_embed_optimizer']  = self.cnn,  self.cnn_optimizer, self.img_embed_to_goal_space, self.lang_embed_to_goal_space_optimizer
        if self.args.gripper_images: saved_objects['gripper_cnn'], saved_objects['griper_cnn_optimizer'] = self.gripper_cnn, self.gripper_cnn_optimizer
        if self.args.use_language: saved_objects['lang_goal_embed'], saved_objects['lang_goal_embed_optimizer'] = self.lang_embed_to_goal_space, self.lang_embed_to_goal_space_optimizer
        return saved_objects


    def save_weights(self, path, run_id=None, experiment_key=None):

        if self.chkpt_manager is None:
            ckpt = tf.train.Checkpoint(**self.get_saved_objects())
            self.chkpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
            save_path = self.chkpt_manager.save()
        else:
            save_path = self.chkpt_manager.save()


    def load_weights(self, path, with_optimizer=False, from_checkpoint=False):
        # With checkpoint
        ckpt = tf.train.Checkpoint(**self.get_saved_objects())
        self.chkpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        ckpt.restore(tf.train.latest_checkpoint(path))

def train_setup(args, dl, GLOBAL_BATCH_SIZE, strategy):
    
    model_params = {'obs_dim':args.img_embedding_size + dl.proprioceptive_features_dim if args.images else dl.obs_dim,
                'goal_dim':args.goal_space_dim if args.images else dl.goal_dim,
                'act_dim':dl.act_dim,
                'layer_size':args.actor_layer_size, 
                'latent_dim':args.latent_dim}



    if args.gripper_images: # separate this from args.images because pybullet sim doens't have a gripper cam in the collected data
        model_params['obs_dim'] += args.gripper_img_embedding_size 


    

    if args.gcbc:
        encoder = None
        planner = None
    else:
        model_params['layer_size'] = args.encoder_layer_size
        if args.encode_all:
            model_params['enc_in_dim'] = model_params['obs_dim'] + model_params['act_dim']
        else:
            model_params['enc_in_dim'] = args.img_embedding_size
        if args.discrete:
          encoder = lfp.model.create_discrete_encoder(**model_params)
        else:
          encoder = lfp.model.create_encoder(**model_params)
          
        model_params['layer_size'] = args.planner_layer_size
        planner = lfp.model.create_planner(**model_params)

    if args.discrete:
        model_params['latent_dim'] = args.latent_dim * args.vq_tiles # there will be a number of tiles

    actor = lfp.model.create_actor(**model_params, gcbc=args.gcbc, num_distribs=args.num_distribs, qbits=args.qbits)

    cnn, gripper_cnn, img_embed_to_goal_space, lang_embed_to_goal_space = None, None, None, None
    if args.images:
        cnn = lfp.model.CNN_DICT[args.cnn_type](dl.img_size, dl.img_size, embedding_size=args.img_embedding_size)
        lfp.utils.build_cnn(cnn)  # Have to do this becasue it is subclassed and the reshapes in the spatial softmax don't play nice with model auto build
        if args.gripper_images:
            gripper_cnn = lfp.model.CNN_DICT[args.cnn_type](dl.gripper_img_size, dl.gripper_img_size, embedding_size=args.gripper_img_embedding_size)
            lfp.utils.build_cnn(gripper_cnn)  # Have to do this becasue it is subclassed and the reshapes in the spatial softmax don't play nice with model auto build
        img_embed_to_goal_space = lfp.model.create_goal_space_mapper(args.img_embedding_size, args.goal_space_dim, args.goal_mapper_layer_size)
        if args.use_language:
            lang_embed_to_goal_space = lfp.model.create_goal_space_mapper(args.sentence_embedding_size, args.goal_space_dim, args.goal_mapper_layer_size)
      

    #optimizer = tfa.optimizers.LAMB(learning_rate=args.learning_rate)
    optimizer = optimizer = tf.optimizers.Adam
    trainer = LFPTrainer(args, actor, dl, encoder, planner, cnn, gripper_cnn, img_embed_to_goal_space, lang_embed_to_goal_space, optimizer, strategy, GLOBAL_BATCH_SIZE)
    return actor, encoder, planner, cnn, gripper_cnn,  img_embed_to_goal_space, lang_embed_to_goal_space, trainer