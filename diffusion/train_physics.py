import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import energyflow as ef
from GSGM import GSGM
import sys


sys.path.append('../')
import utils


train_gen, train_sim, test_gen,test_sim = utils.DataLoader('Pythia26',json_path = '../JSON')
_, _, test_gen_herwig,test_sim_herwig = utils.DataLoader('Herwig',json_path = '../JSON') #Load herwig samples just for evaluation

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
ITERS = len(train_gen)//BATCH_SIZE
model = GSGM()
#Prepare the tf.dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_gen, train_sim))
train_dataset = train_dataset.shuffle(buffer_size=500000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()



lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=LR,
    decay_steps=EPOCHS*ITERS,
)
opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)        
model.compile(            
    optimizer=opt,
    #run_eagerly=True,
    experimental_run_tf_function=False,
    weighted_metrics=[])        

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=ITERS,
    verbose=1,
)

#pythia
detector = tf.constant(test_sim, dtype=tf.float32)
unfold_pythia = model.generate(detector)
np.save("diffusion_Pythia26",unfold_pythia)
#herwig
detector = tf.constant(test_sim_herwig, dtype=tf.float32)
unfold_herwig = model.generate(detector)
np.save("diffusion_Herwig",unfold_herwig)

    
