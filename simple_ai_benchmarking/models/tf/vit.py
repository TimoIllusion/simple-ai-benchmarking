# Project Name: simple-ai-benchmarking
# File Name: vit.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Conversion to TensorFlow of # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# Using 1:1 amount of parameters (85875556)

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model


class MLPBlock(layers.Layer):
    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super(MLPBlock, self).__init__()
        self.fc1 = layers.Dense(mlp_dim, activation=tf.keras.activations.gelu)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(in_dim)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)


class EncoderBlock(layers.Layer):
    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
    ):
        super(EncoderBlock, self).__init__()
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=attention_dropout_rate,
        )  # in keras/tf, the key_dim is not distributed among heads in contrast to pytorch!!
        self.dropout = layers.Dropout(dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout(attn_output)
        out1 = self.ln_1(inputs + attn_output)
        mlp_output = self.mlp(out1)
        return self.ln_2(out1 + mlp_output)


class Encoder(layers.Layer):
    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
    ):
        super(Encoder, self).__init__()
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, seq_length, hidden_dim),
            initializer="random_normal",
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.encoder_layers = [
            EncoderBlock(
                num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_embedding = self.pos_embedding[:, :seq_len, :]
        print("Before PE (TF):", inputs.shape)

        x = inputs + pos_embedding
        print("after PE (TF):", x.shape)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return self.ln(x)


class ConcatClassTokenLayer(layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(ConcatClassTokenLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        broadcast_class_token = tf.broadcast_to(
            self.class_token, (batch_size, 1, self.hidden_dim)
        )

        return tf.concat([broadcast_class_token, inputs], axis=1)

    def get_config(self):
        config = super(ConcatClassTokenLayer, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class VisionTransformer(Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        num_classes,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    ):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size

        self.conv_proj = tf.keras.layers.Conv2D(
            hidden_dim, kernel_size=patch_size, strides=patch_size
        )

        num_patches = (image_size // patch_size) ** 2  # is seq_length

        self.class_token_concatenation = ConcatClassTokenLayer(hidden_dim)

        num_patches += 1  # account for the additional class token

        self.encoder = Encoder(
            num_patches,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
        )

        self.mlp_head = layers.Dense(num_classes)

    def call(self, inputs):

        print("Start (TF):", inputs.shape)
        x = self.conv_proj(inputs)
        print("After projection (TF):", x.shape)

        batch_size = tf.shape(inputs)[0]

        x = tf.reshape(
            x, (batch_size, -1, x.shape[-1])
        )  # (batch_size, num_patches, hidden_dim)

        ## Theis code of torchvision vit has bee transfered to ConcatClassTokenLayer
        # batch_class_token = tf.broadcast_to(
        #     self.class_token, (batch_size, 1, self.class_token.shape[-1])
        # )
        # x = tf.concat([batch_class_token, x], axis=1)
        x = self.class_token_concatenation(x)

        x = self.encoder(x)

        # Take the output corresponding to the class token
        x = x[:, 0]

        x = self.mlp_head(x)

        return x


def create_vit_b_16(num_classes: int, model_sample_shape: Tuple[int]) -> VisionTransformer:

    assert len(model_sample_shape) == 3, "Expecting image shape for the model sample (H, W, C)"
    assert model_sample_shape[0] == model_sample_shape[1], "Expecting square image shape for the model sample"
    
    height, width, channels = model_sample_shape

    image_size = height
    
    # ViT-B/16 configuration
    patch_size = 16
    num_layers = 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 3072

    dropout_rate = 0.1
    attention_dropout_rate = 0.1

    vit_b_16_model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
    )

    vit_b_16_model.build(
        input_shape=(None, image_size, image_size, 3)
    )

    return vit_b_16_model


if __name__ == "__main__":
    vit_b_16_model = create_vit_b_16(100, (224, 224, 3))

    # compile model
    vit_b_16_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
    )

    sample_input = tf.zeros([1, 224, 224, 3])
    _ = vit_b_16_model(sample_input)

    save_path = "saved_model_vit_b_16"
    vit_b_16_model.save(save_path, save_format="tf")
