"""Base Invertible Module Class"""

from typing import Tuple, Iterable
import tensorflow as tf


class InvertibleModule(tf.keras.layers.Layer):
    """
    Generic invertible layer for invertible
    neural network structures.

    Used to implement `Nice`, `Glow`, `RNVP` etc.
    """

    def __init__(self, dims_in: Tuple[int], dims_c: Iterable[Tuple[int]] = None):
        """
        Args:
            dims_in: a tuple specifying the shape of the input,
                     excluding the batch dimension, to this
                     operator: ``dims_in = (dim_0,..., channels)``
            dims_c:  a list of tuples specifying the shape
                     of the conditions to this operator,
                     excluding the batch dimension

        ** Note  to implementors:**

        - The shapes are in the standard TensorFlow 'channels_last' format.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = tuple(dims_in)
        self.dims_c = list(dims_c)

    def call(  # pylint: disable=W0221
        self,
        x_or_z: tf.Tensor,
        c: Iterable[tf.Tensor] = None,
        rev: bool = False,
        jac: bool = True,
    ):
        """
        Perform a forward (default, ``rev=False``) or
        backward pass (``rev=True``) through this layer.
        Args:
            x_or_z: input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide call(...) method"
        )

    def get_config(self):
        "Needed within TensorFlow to serialize this layer"
        config = {"dims_in": self.dims_in, "dims_c": self.dims_c}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
