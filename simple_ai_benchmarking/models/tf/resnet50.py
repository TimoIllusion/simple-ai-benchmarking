from typing import Tuple

import tensorflow as tf


class ResNet50(tf.keras.Model):

    def __init__(self, num_classes: int, input_shape: Tuple[int]):
        super(ResNet50, self).__init__()
        self.resnet50 = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes,
        )

    def call(self, x):
        return self.resnet50(x)
