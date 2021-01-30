import umap
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import io

reducer = umap.UMAP(metric='cosine', random_state=42)

colors_dict = {
 'press button': [1,0,0], # red
 'dial off': [0.2, 0.6, 0.2], # green
 'dial on': [0, 0.8, 0], # bolder green
 'door left':[0.6, 0, 0.8], # purple #9900cc
 'door right':[0.8,0.2,0.99],
 'drawer in':[0.4, 0.2,0], # browns #cc9900
 'drawer out':[0.8, 0.6, 0],
 'block in drawer':[0.9,0.9,0.4], # yellows
 'block out of drawer':[1,0.8,0.8],
 'block in cupboard left':[1,0.4,1], #brighter purple #ff66ff
 'block in cupboard right':[0.8, 0.4, 1.0], # lighter purpple #cc33ff,,
 'block out of cupboard left':[0.6, 0.4, 1.0], # bluer purpe #9966ff
 'block out of cupboard right':[0.6, 0.6, 1.0], # bluer lilac #9999ff
 'pick place': [0, 0.7, 1.0], # #00BFFF eveertything from here down is shades of blue
 'knock down block': [0.1, 0.6, 0.8],
 'stand up block': [0.05, 0.4, 0.8], ##0E61D1
 'lift up': [0.05, 0.2, 0.7], # #1432BA
 'take down':[0.03, 0.001, 0.6], # #1B03A3
 'rotate block left': [0.2, 0.8, 0.8],
 'rotate block right':[0.2, 0.8, 1.0] 
}


bucket_map = {
 'press button': 'button',
 'dial off': 'dial',
 'dial on': 'dial',
 'door left':'cupboard door left',
 'door right':'cupboard door right',
 'drawer in':'drawer in',
 'drawer out':'drawer out',
 'block in drawer':'block in/out drawer',
 'block out of drawer':'block in/out drawer',
 'block in cupboard left':'block in/out cupboard',
 'block in cupboard right':'block in/out cupboard',
 'block out of cupboard left':'block in/out cupboard',
 'block out of cupboard right':'block in/out cupboard',
 'pick place': 'block',
 'knock down block': 'block',
 'stand up block': 'block', ##0E61D1
 'lift up': 'block + shelf',
 'take down': 'block + shelf',
 'rotate block left': 'block',
 'rotate block right':'block', 
}

bucket_colors = {
 'button': [1,0,0],
 'dial': [0.6, 0, 0.8],
 'cupboard door left':[0.2, 0.6, 0.2] , # purple #9900cc
 'cupboard door right':[0.4, 0.9, 0.2] , # purple #9900cc
 'drawer in':[0.4, 0.2,0], # browns #cc9900
 'drawer out':[0.8, 0.6, 0],
 'block in/out drawer':[0.8,0.8,0.4], # yellows
 'block in/out cupboard':[0.8, 0.4, 1.0], #brighter purple #ff66ff
 'block': [0, 0.7, 1.0], # #00BFFF eveertything from here down is shades of blue
 'block + shelf': [0.05, 0.2, 0.7],
}



hold_out = ['dial']

def get_labelled_trajs(TEST_DATA_PATH, bucket=False):
    test_labels = np.load(TEST_DATA_PATH/'trajectory_labels.npz', allow_pickle=True)['trajectory_labels']
    acts,obs, goals, labels, colors, paths = [], [], [], [], [], []
    # this could be sped up significantly by just storing the trajs in memory, it takes ms on my local, but is a bit slow with colabs cpu
    

    for k,v in test_labels.flatten()[0].items():
      if bucket:
        if v in bucket_map:
          v = bucket_map[v]
          c = bucket_colors[v]
      else:
        c = colors_dict[v]
      if v not in hold_out:
        folder = k.split('states_and_ims/')[1].split('/')[0]
        start = int(k.split('env_states/')[1].split('/')[0].strip('.bullet'))
        data = np.load(TEST_DATA_PATH/'obs_act_etc/'/folder/'data.npz')
        traj_len = 40
        end = start + traj_len #min(len(data['acts_rpy'])-1,start+traj_len )
        traj_acts = data['acts'][start:end]
        traj_obs = data['obs'][start:end]
        traj_goals = data['achieved_goals'][end]
        acts.append(traj_acts), obs.append(traj_obs), goals.append(traj_goals), labels.append(v), colors.append(c), paths.append(TEST_DATA_PATH/"/".join(k.split('/')[4:]))

    return np.array(obs), np.array(acts), np.array(goals), labels, colors, paths

def project_labelled_latents(reducer, z_enc, colors, bucket=True):
    reducer.fit(z_enc)
    z_enc_embed = reducer.transform(z_enc)
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(z_enc_embed[:, 0], z_enc_embed[:, 1], s=60, label='z_enc', c = colors)
    ax.set_aspect('equal', 'datalim')
    ax.legend(loc='upper left')
    #plt.axis('off')
    plt.tight_layout()
    
    # The following two lines generate custom fake lines that will be used as legend entries:
    if bucket:
      colors_dict = bucket_colors.copy()
    for i in hold_out:
      del colors_dict[i]
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors_dict.values()]
    plt.legend(markers, colors_dict.keys(), numpoints=1)
    return fig,scatter


def produce_cluster_fig(batch,encoder,planner,TEST_DATA_PATH, num_take):
    bucket = True
    obs, acts, goals, labels, colors, paths = get_labelled_trajs(TEST_DATA_PATH, bucket=bucket)
    batch_states,batch_acts, batch_goals, batch_colors = batch['obs'][:num_take, :40, :],batch['acts'][:num_take, :40, :], batch['goals'][:num_take, 0, :], [[0.8,0.8,0.8,0.6]]*num_take
    obs  = np.concatenate([obs, batch_states])
    acts = np.concatenate([acts, batch_acts])
    goals = np.concatenate([goals, batch_goals])
    initial_state = obs[:, 0, :]
    z_enc = encoder((obs,acts))
    z_plan = planner((initial_state, goals))
    fig_enc, scatter = project_labelled_latents(reducer, z_enc.sample(), colors + batch_colors, bucket)
    fig_plan, scatter = project_labelled_latents(reducer, z_plan.sample(), colors + batch_colors, bucket)
    
    return fig_enc, fig_plan



def project_enc_and_plan(z_enc, z_plan, connecting_lines=False):
    ''' Overlays encoder and planner '''
    
    z_combined = tf.concat([z_enc, z_plan], axis = 0)
    reducer.fit(z_combined)
    l = len(z_enc)
    z_embed = reducer.transform(z_combined)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    z_enc = z_embed[:l]
    z_plan = z_embed[l:]
    if connecting_lines:
        for i in range(len(z_enc)): plt.plot([z_enc[i,0],z_plan[i,0]], [z_enc[i,1],z_plan[i,1]], c='grey', linewidth = 0.5)
    ax.scatter(z_enc[:,0], z_enc[:,1], s=5, label='z_enc')
    ax.scatter(z_plan[:,0], z_plan[:, 1], s=5, label='z_plan')
    
    ax.set_aspect('equal', 'datalim')
    ax.legend(loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image