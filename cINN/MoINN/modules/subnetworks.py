""" Subnetworks """

from typing import Dict

import tensorflow as tf


# pylint: disable=C0103
class _SubNet(tf.keras.layers.Layer):
    """Base class to implement various subnetworks.  It takes care of
    checking the dimensions. Each child class only has
    to implement the _network() method.
    """

    def __init__(self, meta: Dict, channels_in: int, channels_out: int):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          channels_in_in:
            Number of input channels.
          channels_in_out:
            Number of output channels.
        """
        super().__init__()
        self.meta = meta
        self.channels_in = channels_in
        self.channels_out = channels_out

    def call(self, x):  # pylint: disable=W0221
        """
        Perform a forward pass through this layer.
        Args:
          x: input data (array-like of one or more tensors)
            of the form: ``x = input_tensor_1``.
        """
        out = self._network(x)
        return out

    def _network(self, x):
        """The network operation used in the call() function.
        Args:
          x (Tensor): the input tensor.
        Returns:
          y (Tensor): the output tensor.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _network(...) method"
        )

    def build(self, input_shape):
        """
        Helps to prevent wrong usage of modules and helps for debugging.
        """
        assert (
            input_shape[-1] == self.channels_in
        ), f"Channel dimension of input ({input_shape[-1]}) and given input channels ({self.channels_in}) don't agree."

        super().build(input_shape)

    def get_config(self):
        config = {
            "meta": self.meta,
            "channels_in": self.channels_in,
            "channels_out": self.channels_out,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseSubNet(_SubNet):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(self, meta, channels_in, channels_out):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          channels_in_in:
            Number of input channels.
          channels_in_out:
            Number of output channels.
        """
        super().__init__(meta, channels_in, channels_out)

        # which activation
        if isinstance(meta["activation"], str):
            if meta["activation"] == "relu":
                activation = tf.keras.activations.relu
            elif meta["activation"] == "elu":
                activation = tf.keras.activations.elu
            elif meta["activation"] == "leakyrelu":
                activation = tf.keras.layers.LeakyReLU()
            elif meta["activation"] == "tanh":
                activation = tf.keras.activations.tanh
            else:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            activation = meta["activation"]

        # Define the layers
        self.hidden_layers = [
            tf.keras.layers.Dense(
                self.meta["units"],
                activation=activation,
                kernel_initializer=self.meta["initializer"],
            )
            for i in range(self.meta["layers"])
        ]

        self.dense_out = tf.keras.layers.Dense(
            self.channels_out, kernel_initializer=self.meta["initializer"]
        )

    def _network(self, x):
        """The used layers in this Subnetwork.
        Returns:
          _layers (tf.keras.layers): Some stacked keras layers.
        """
        for layer in self.hidden_layers:
            x = layer(x)

        y = self.dense_out(x)
        return y


class Conv1DSubNet(_SubNet):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(self, meta, channels_in, channels_out):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          channels_in_in:
            Number of input channels.
          channels_in_out:
            Number of output channels.
        """
        super().__init__(meta, channels_in, channels_out)

        # which activation
        if isinstance(meta["activation"], str):
            if meta["activation"] == "relu":
                activation = tf.keras.activations.relu
            elif meta["activation"] == "elu":
                activation = tf.keras.activations.elu
            elif meta["activation"] == "leakyrelu":
                activation = tf.keras.layers.LeakyReLU()
            elif meta["activation"] == "tanh":
                activation = tf.keras.activations.tanh
            else:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            activation = meta["activation"]

        # Define the layers
        self.hidden_layers = [
            tf.keras.layers.Conv1D(
                self.meta["filters"],
                self.meta["kernel_size"],
                padding="same",
                activation=activation,
                kernel_initializer=self.meta["initializer"],
            )
            for i in range(self.meta["layers"])
        ]

        self.conv_out = tf.keras.layers.Conv1D(
            self.channels_out,
            1,
            padding="same",
            kernel_initializer=self.meta["initializer"],
        )

    def build(self, input_shape):
        """
        Helps to prevent wrong usage of modules and helps for debugging.
        """
        assert (
            len(input_shape) == 3
        ), f"Dimension of input ({len(input_shape)-1}D) and dimension (2D) don't agree"

        super().build(input_shape)

    def _network(self, x):
        """The used layers in this Subnetwork.
        Returns:
          _layers (tf.keras.layers): Some stacked keras layers.
        """
        for layer in self.hidden_layers:
            x = layer(x)

        y = self.conv_out(x)
        return y
