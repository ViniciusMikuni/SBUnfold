import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
from deepsets import Resnet
import gc

#tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM'):
        super(GSGM, self).__init__()

        self.activation = layers.LeakyReLU(alpha=0.01)                
        self.num_input = 6
        
        self.num_embed = 16
        self.ema=0.999
        self.minlogsnr = -20.0
        self.maxlogsnr = 20.0
        self.num_steps = 1000
        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs = Input((self.num_input))
        inputs_cond = Input((self.num_input))

        time_conditional = self.Embedding(inputs_time,self.projection)
        reco_conditional = layers.Dense(self.num_embed,activation=None)(inputs_cond) 
        reco_conditional = self.activation(reco_conditional)
        input_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [time_conditional,reco_conditional],-1))
        input_conditional=self.activation(input_conditional)
                        
        outputs = Resnet(
            inputs,
            self.num_input,
            input_conditional,
            num_embed=self.num_embed,
            num_layer = 4,
            mlp_dim= 32,
        )

        self.model = keras.Model(inputs=[inputs,inputs_time,inputs_cond],
                                 outputs=outputs)

        print(self.model.summary())
        self.ema_model = keras.models.clone_model(self.model)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq


    def Embedding(self,inputs,projection):
        angle = inputs*projection*1000
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)

    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min, logsnr_max):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))
    
    @tf.function
    def get_logsnr_alpha_sigma(self,time,shape=None):
        logsnr = self.logsnr_schedule_cosine(time,logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        if shape is not None:
            alpha = tf.reshape(alpha,shape)
            sigma = tf.reshape(sigma,shape)
        return logsnr, alpha, sigma

    @tf.function
    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min, logsnr_max):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a
            
    
    @tf.function
    def train_step(self, inputs):
        eps=1e-5        
        x,cond = inputs            
        random_t = tf.random.uniform((tf.shape(x)[0],1))            
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)

        with tf.GradientTape() as tape:
            z = tf.random.normal((tf.shape(x)),dtype=tf.float32)
            
            perturbed_x = alpha*x + z * sigma
            pred = self.model([perturbed_x,random_t,cond])
            v = alpha* z - sigma* x
            losses = tf.square(pred - v)                
            loss = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
 
        trainable_variables = self.model.trainable_variables
        g = tape.gradient(loss, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))        
        self.loss_tracker.update_state(loss)

            
        for weight, ema_weight in zip(self.model.weights, self.ema_model.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {
            "loss": self.loss_tracker.result(), 
        }


    @tf.function
    def test_step(self, inputs):
        eps=1e-5        
        x,cond = inputs            
        random_t = tf.random.uniform((tf.shape(x)[0],1))            
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)


        z = tf.random.normal((tf.shape(x)),dtype=tf.float32)
        
        perturbed_x = alpha*x + z * sigma
        pred = self.model([perturbed_x,random_t,cond])
        v = alpha* z - sigma* x
        losses = tf.square(pred - v)                
        loss = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(), 
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)

    def generate(self,cond):
        start = time.time()
        samples = self.DDPMSampler(cond,self.ema_model,
                                   data_shape=[cond.shape[0],self.num_input],
                                   const_shape = (-1,1)).numpy()            
        return samples
    

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)

        
        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)
            
            v = model([x, random_t,cond],training=False)                            
            mean = alpha * x - sigma * v
            eps = sigma * x + alpha * v            
            x = alpha_ * mean + sigma_ * eps
            
        # The last step does not include any noise
        return mean        




