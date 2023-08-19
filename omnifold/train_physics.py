import tensorflow as tf
import numpy as np
import sys
import time

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

sys.path.append('../')
import utils

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

N_t = 1000
# import data
train_gen, train_sim, test_gen,test_sim = utils.DataLoader('Pythia26',json_path = '../JSON')
train_gen_herwig, train_sim_herwig, test_gen_herwig,test_sim_herwig = utils.DataLoader('Herwig',N_t = N_t,json_path = '../JSON') #Load herwig samples just for evaluation


#Define the classifier model
inputs = Input((train_sim.shape[1], ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(150, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss=weighted_binary_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])

X_train = np.concatenate((train_sim, train_sim_herwig))
Y_train = np.concatenate((np.zeros(train_sim.shape[0]), np.ones(train_sim_herwig.shape[0])))
#Initial weights
weights = np.concatenate((np.ones(train_sim.shape[0]), float(train_sim.shape[0]/train_sim_herwig.shape[0])*np.ones(train_sim_herwig.shape[0])))
print(weights[0],weights[-1])


Y_train = np.stack((Y_train, weights), axis=1)

# train the network
EPOCHS = 30
BATCH_SIZE = 128

model.fit(X_train,
          Y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=True)


weights = reweight(test_sim,model)
np.save("omnifold_Pythia26_{}".format(N_t),test_gen)
np.save("weights_data_{}".format(N_t),weights)
