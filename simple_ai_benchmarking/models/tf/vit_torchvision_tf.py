# tf conversion of # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

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
            num_heads=num_heads, key_dim=hidden_dim // num_heads, dropout=attention_dropout_rate
        ) # in keras/tf, the key_dim is not distributed among heads in contrast to pytorch!!
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
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
        seq_length,
    ):
        super(Encoder, self).__init__()
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=(1, seq_length + 1, hidden_dim),
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
        x = inputs + pos_embedding
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return self.ln(x)

#TODO: fix weights amount, currently 768 too many params
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
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.conv_proj = tf.keras.layers.Conv2D(
            hidden_dim, kernel_size=patch_size, strides=patch_size
        )
        self.class_token = self.add_weight(
            "class_token", shape=(1, 1, hidden_dim), initializer="zeros"
        )
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            num_patches + 1,
        )
        self.mlp_head = layers.Dense(num_classes)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        print("Start (TF):", inputs.shape)
        x = self.conv_proj(inputs)
        print("After projection (TF):", x.shape)
        x = tf.reshape(
            x, (batch_size, -1, x.shape[-1])
        )  # (batch_size, num_patches, hidden_dim)
        class_token = tf.broadcast_to(
            self.class_token, (batch_size, 1, self.class_token.shape[-1])
        )
        # # Concatenate the class token with the patch embeddings
        x = tf.concat([class_token, x], axis=1)
        
        print("Before encoder (TF):", x.shape)

        # Pass through the encoder
        x = self.encoder(x)

        print("After encoder (TF):", x.shape)
        # Take the output corresponding to the class token
        class_token_final = x[:, 0]

        print("Before head (TF):", class_token_final.shape)
        # Pass the class token through the final MLP head to get the predictions
        output = self.mlp_head(class_token_final)
        print("After head (TF):", output.shape)
        
        return output


def create_vit_b_16() -> VisionTransformer:

    # Define the VisionTransformer model parameters as per the ViT-B/16 configuration
    image_size = 224  # Assuming input images are 224x224
    patch_size = 16
    num_layers = 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 3072
    num_classes = 100  # Assuming the model targets the 1000-class ImageNet dataset
    dropout_rate = 0.1
    attention_dropout_rate = 0.1

    # Instantiate the VisionTransformer model with the specified parameters
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

    # Display the model summary to verify the architecture
    vit_b_16_model.build(
        input_shape=(None, image_size, image_size, 3)
    )  # Build the model with the input shape (batch_size, height, width, channels)
    # vit_b_16_model.summary()
    return vit_b_16_model


if __name__ == "__main__":
    vit_b_16_model = create_vit_b_16()
    
    # compile model
    vit_b_16_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # To use target shape of (N, ) instead of (N, num_classes)
        metrics=["accuracy"],
    )
    
    sample_input = tf.zeros([1, 224, 224, 3])
    _ = vit_b_16_model(sample_input)

    # Save the model using tf.saved_model.save
    save_path = "saved_model_vit_b_16"
    vit_b_16_model.save(save_path, save_format="tf")
