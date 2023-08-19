import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import time
import energyflow as ef
from sklearn.preprocessing import StandardScaler

from cflow import ConditionalFlow
from MoINN.modules.subnetworks import DenseSubNet

from util import train_density_estimation, plot_loss, plot_tau_ratio

sys.path.append('../')
import utils


# import data

train_gen, train_sim, test_gen,test_sim = utils.DataLoader('Pythia26',json_path = '../JSON')
_, _, test_gen_herwig,test_sim_herwig = utils.DataLoader('Herwig',json_path = '../JSON') #Load herwig samples just for evaluation


# Get the flow
meta = {
        "units": 16,
        "layers": 4,
        "initializer": "glorot_uniform",
        "activation": "leakyrelu",
        }

cflow = ConditionalFlow(dims_in=[6], dims_c=[[6]], n_blocks=12,
                        subnet_meta=meta, subnet_constructor=DenseSubNet)

# train the network
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
DECAY_RATE=0.1
ITERS = len(train_gen)//BATCH_SIZE
DECAY_STEP=ITERS

#Prepare the tf.dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_gen, train_sim))
train_dataset = train_dataset.shuffle(buffer_size=500000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

train_losses = []
#train_all = np.concatenate([train_gen, train_sim], axis=-1)
start_time = time.time()
for e in range(EPOCHS):
    
    batch_train_losses = []
    # Iterate over the batches of the dataset.
    for step, (batch_gen, batch_sim) in enumerate(train_dataset):
        batch_loss = train_density_estimation(cflow, opt, batch_gen, [batch_sim])
        batch_train_losses.append(batch_loss)
        

    train_loss = tf.reduce_mean(batch_train_losses)
    train_losses.append(train_loss)

    if (e + 1) % 1 == 0:
        # Print metrics
        print(
            "Epoch #{}: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_losses[-1], opt._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time)/60/60))
print("--- Run time: %s mins ---" % ((end_time - start_time)/60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))


# Make plots and sample
#plot_loss(train_losses, name="Log-likelihood", log_axis=False)

detector = tf.constant(test_sim, dtype=tf.float32)
unfold_pythia = cflow.sample(test_sim.shape[0],[detector])
np.save("inn_Pythia26",unfold_pythia)

#herwig

detector = tf.constant(test_sim_herwig, dtype=tf.float32)
unfold_pythia = cflow.sample(test_sim_herwig.shape[0],[detector])
np.save("inn_Herwig",unfold_pythia)
