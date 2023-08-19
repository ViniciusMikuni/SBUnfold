import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import importlib
import math
import numpy as np
import sys
import tensorflow as tf
import time
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Union

# Import Model blocks
from MoINN.modules.all_in_one_block import AllInOneBlock
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import matplotlib.pyplot as plt

class ConditionalFlow(tf.keras.Model):
    """Defines the conditional flow network"""

    def __init__(
        self,
        dims_in,
        dims_c,
        n_blocks,
        subnet_meta: Dict = None,
        subnet_constructor: callable = None,
        name="cflow",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dims_in = dims_in
        self.dims_c = dims_c
        self.sum_dims = tuple(range(1, 1 + len(self.dims_in)))

        layer_list = []
        for i in range(n_blocks):
            layer_list.append(AllInOneBlock(dims_in, dims_c=dims_c, permute_soft=True, subnet_meta=subnet_meta, subnet_constructor=subnet_constructor))
        self.layer_list = layer_list

    def distribution_log_prob(self, inputs):
        log_z = tf.constant(0.5 * np.prod(self.dims_in) * np.log(2 * np.pi), dtype=tf.float32)
        neg_energy = -0.5 * tf.reduce_sum(inputs ** 2, axis =self.sum_dims)
        return neg_energy - log_z

    def call(self, x, c=None):
        y = x
        log_det = 0
        for layer in self.layer_list:
            y, ljd = layer(y, c=c)
            log_det += ljd
        return y, log_det

    def log_prob(self, x, context=None):
        noise, logabsdet = self.call(x, c=context)
        log_prob = self.distribution_log_prob(noise)
        return log_prob + logabsdet

    def sample(self, n_samples, c=None):
        shape = (n_samples,) + tuple(self.dims_c[0])
        z = tf.random.normal(shape)
        y = z
        for layer in self.layer_list[::-1]:
            y = layer(y, c=c, rev=True, jac=False)

        return y

    def prob_sample(self, n_samples, unfoldings, c=None):
        shape = (n_samples,) + tuple(self.dims_c[0])

        output = []
        for _ in range(unfoldings):
            z = tf.random.normal(shape)
            y = z
            for layer in self.layer_list[::-1]:
                y = layer(y, c=c, rev=True, jac=False)
            y = tf.expand_dims(y, axis=-1)
            output.append(y)
        
        sample = tf.concat(output, axis=-1)
        return sample
