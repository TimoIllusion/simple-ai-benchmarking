import tensorflow as tf
from tensorflow.keras import layers


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_layers = [
            layers.Dense(units, activation=tf.nn.gelu) for units in hidden_units
        ]
        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in hidden_units]

    def call(self, inputs):
        x = inputs
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        return x


class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, transformer_units, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(hidden_units=transformer_units, dropout_rate=dropout_rate)

    def call(self, inputs):
        x1 = self.norm1(inputs)
        attn_output = self.mha(x1, x1)
        x2 = layers.Add()([attn_output, inputs])  # Skip connection
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return layers.Add()([x3, x2])  # Skip connection


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units,
        num_classes,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes

        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_blocks = [
            TransformerBlock(projection_dim, num_heads, transformer_units, 0.1)
            for _ in range(transformer_layers)
        ]
        self.flatten = layers.Flatten()
        self.final_mlp = MLP(
            hidden_units=mlp_head_units + [num_classes], dropout_rate=0.5
        )

    def call(self, inputs):
        x = self.patches(inputs)
        x = self.patch_encoder(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.flatten(x)
        return self.final_mlp(x)
