import tensorflow as tf

class MaxMetric(tf.keras.metrics.Metric):

    def __init__(self, name='max_metric', **kwargs):
        super(MaxMetric, self).__init__(name=name, **kwargs)
        self.max = self.add_weight(name='max', initializer='zeros')
        self.abs_err = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def update_state(self, y_true, y_pred, mask=1.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        values = self.abs_err(y_true, y_pred) * mask
        values = tf.cast(values, self.dtype)
        max = tf.reduce_max(values)
        if tf.math.greater(max, self.max):
            self.max.assign(max)

    def result(self):
        return self.max
    
    
    
def record(value, metric):
    metric.update_state(value)
    return value


def log(metric):
    result = metric.result()
    metric.reset_states()
    return result
    
def create_metrics():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    actor_grad_norm = tf.keras.metrics.Mean(name='actor_grad_norm')
    encoder_grad_norm = tf.keras.metrics.Mean(name='encoder_grad_norm')
    planner_grad_norm = tf.keras.metrics.Mean(name='planner_grad_norm')

    actor_grad_norm_clipped = tf.keras.metrics.Mean(name='actor_grad_norm_clipped')
    encoder_grad_norm_clipped = tf.keras.metrics.Mean(name='encoder_grad_norm_clipped')
    planner_grad_norm_clipped = tf.keras.metrics.Mean(name='planner_grad_norm_clipped')

    global_grad_norm = tf.keras.metrics.Mean(name='global_grad_norm')

    test = tf.keras.metrics.Mean(name='test')
    test2 = tf.keras.metrics.Mean(name='test2')

    train_act_with_enc_loss = tf.keras.metrics.Mean(name='train_act_with_enc_loss')
    train_act_with_plan_loss = tf.keras.metrics.Mean(name='train_act_with_plan_loss')
    valid_act_with_enc_loss = tf.keras.metrics.Mean(name='valid_act_with_enc_loss')
    valid_act_with_plan_loss = tf.keras.metrics.Mean(name='valid_act_with_plan_loss')

    train_reg_loss = tf.keras.metrics.Mean(name='reg_loss')
    valid_reg_loss = tf.keras.metrics.Mean(name='valid_reg_loss')

    valid_position_loss = tf.keras.metrics.Mean(name='valid_position_loss')
    valid_max_position_loss = MaxMetric(name='valid_max_position_loss')
    valid_rotation_loss = tf.keras.metrics.Mean(name='valid_rotation_loss')
    valid_max_rotation_loss = MaxMetric(name='valid_max_rotation_loss')
    valid_gripper_loss = tf.keras.metrics.Mean(name='valid_rotation_loss')
    return train_loss, valid_loss, actor_grad_norm, encoder_grad_norm, planner_grad_norm, \
          actor_grad_norm_clipped, encoder_grad_norm_clipped, planner_grad_norm_clipped, global_grad_norm, \
          test, test2,  train_act_with_enc_loss, train_act_with_plan_loss, valid_act_with_enc_loss, valid_act_with_plan_loss,\
          train_reg_loss, valid_reg_loss, valid_position_loss, valid_max_position_loss, valid_rotation_loss, valid_max_rotation_loss, valid_gripper_loss



def log_action_breakdown(policy, actions, mask, seq_lens, config, valid_position_loss, valid_max_position_loss, valid_rotation_loss, valid_max_rotation_loss, valid_gripper_loss, compute_MAE):
    if quat_act:
        # xyz, q1-4, grip
        action_breakdown = [3, 4, 1]
    else:
        action_breakdown = [3, 3, 1]

    # pos, rot, gripper individual losses
    if config.num_distribs is not None:
        pos_acts, rot_acts, grip_act = tf.split(policy.sample(), action_breakdown, -1)
    else:
        pos_acts, rot_acts, grip_act = tf.split(policy, action_breakdown, -1)

    # get the breakdown of the true ones
    true_pos_acts, true_rot_acts, true_grip_act = tf.split(actions, action_breakdown, -1)

    valid_position_loss.update_state(compute_MAE(true_pos_acts, pos_acts, mask, seq_lens))
    valid_max_position_loss(true_pos_acts, pos_acts, mask)
    valid_rotation_loss.update_state(compute_MAE(true_rot_acts, rot_acts, mask, seq_lens))
    valid_max_rotation_loss(true_rot_acts, rot_acts, mask)
    valid_gripper_loss.update_state(compute_MAE(true_grip_act, grip_act, mask, seq_lens))
