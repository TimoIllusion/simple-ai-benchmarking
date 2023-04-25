# Modified version of Khalid Salama's "Image classification with modern MLP models" from https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py (Apache License 2.0)
# Changes:
# - remove gMLP and FNet code
# - convert script into class ModernMLPKeras that has three main functions: setup, train and infer
# - turn variable into member variables of class ModernMLPKeras to be able to access them across functions
# - remove/change comments to fit adjusted code
# - adjust hyperparameters to fit quick benchmark
# - split run_experiments function and integrate into setup and train functions
# - add steps per epoch parameter to change size of epochs (to speed up benchmark)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from simple_ai_benchmarking import AIWorkload

class MLPMixer(AIWorkload):
    """
    The [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.
    """
    
    def __init__(self, batch_size: int = 128):
        
        self.mlpmixer_classifier = None
        
        """
        ## Configure the hyperparameters
        """
        
        self.num_classes = 100
        self.input_shape = (32, 32, 3)
        self.learning_rate = 0.005
        self.weight_decay = 0.0001
        self.batch_size = batch_size
        self.num_epochs = 3 # Original: 50
        self.steps_per_epoch=10 # Original: None
        self.dropout_rate = 0.2
        self.image_size = 64  # We'll resize input images to this size.
        self.patch_size = 8  # Size of the patches to be extracted from the input images.
        self.num_patches = (self.image_size // self.patch_size) ** 2  # Size of the data array.
        self.embedding_dim = 256  # Number of hidden units.
        self.num_blocks = 4  # Number of blocks.
        
        
        # Create a learning rate scheduler callback.
        self.reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        )
        # Create an early stopping callback.
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        

        print(f"Image size: {self.image_size} X {self.image_size} = {self.image_size ** 2}")
        print(f"Patch size: {self.patch_size} X {self.patch_size} = {self.patch_size ** 2} ")
        print(f"Patches per image: {self.num_patches}")
        print(f"Elements per patch (3 channels): {(self.patch_size ** 2) * 3}")
        
        
    def _build_classifier(self, blocks, positional_encoding=False):
        """
        ## Build a classification model

        We implement a method that builds a classifier given the processing blocks.
        """

        inputs = layers.Input(shape=self.input_shape)
        # Augment data.
        augmented = self.data_augmentation(inputs)
        # Create patches.
        patches = Patches(self.patch_size, self.num_patches)(augmented)
        # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
        x = layers.Dense(units=self.embedding_dim)(patches)
        if positional_encoding:
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            position_embedding = layers.Embedding(
                input_dim=self.num_patches, output_dim=self.embedding_dim
            )(positions)
            x = x + position_embedding
        # Process x using the module blocks.
        x = blocks(x)
        # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
        representation = layers.GlobalAveragePooling1D()(x)
        # Apply dropout.
        representation = layers.Dropout(rate=self.dropout_rate)(representation)
        # Compute logits outputs.
        logits = layers.Dense(self.num_classes)(representation)
        # Create the Keras model.
        return keras.Model(inputs=inputs, outputs=logits)
    
    def setup(self):


        """
        ## Prepare the data
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar100.load_data()

        print(f"x_train shape: {self.x_train.shape} - y_train shape: {self.y_train.shape}")
        print(f"x_test shape: {self.x_test.shape} - y_test shape: {self.y_test.shape}")


        """
        ## Use data augmentation
        """
        self.data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(self.image_size, self.image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        self.data_augmentation.layers[0].adapt(self.x_train)


        """
        ### Build the MLP-Mixer model
        """

        mlpmixer_blocks = keras.Sequential(
            [MLPMixerLayer(self.num_patches, self.embedding_dim, self.dropout_rate) for _ in range(self.num_blocks)]
        )

        self.mlpmixer_classifier = self._build_classifier(mlpmixer_blocks)
        
        # Create Adam optimizer with weight decay.
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Compile the model.
        self.mlpmixer_classifier.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
            ],
        )
        

        
    def train(self):
        
        assert self.mlpmixer_classifier is not None, "Please run setup method before starting training!"
        
        """
        ### Train the MLP-Mixer model

        Note that training the model with the current settings on a V100 GPUs
        takes around 8 seconds per epoch.
        """
        
        
        # Fit the model.
        history = self.mlpmixer_classifier.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.num_epochs,
            validation_split=0.1,
            callbacks=[self.early_stopping, self.reduce_lr],
        )
        

        """
        The MLP-Mixer model tends to have much less number of parameters compared
        to convolutional and transformer-based models, which leads to less training and
        serving computational cost.

        As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
        when pre-trained on large datasets, or with modern regularization schemes,
        the MLP-Mixer attains competitive scores to state-of-the-art models.
        You can obtain better results by increasing the embedding dimensions,
        increasing, increasing the number of mixer blocks, and training the model for longer.
        You may also try to increase the size of the input images and use different patch sizes.
        """
        
    def eval(self):
        """
        ### Evaluate the MLP-Mixer model
        """
        
        #TODO: use predict method here, or create new function
        
        _, accuracy, top_5_accuracy = self.mlpmixer_classifier.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        
    def build_log_dict(self):
        
        if self.steps_per_epoch is None:
            samples_training = self.num_epochs * len(self.x_train)
        else:
            samples_training = self.num_epochs * self.steps_per_epoch * self.batch_size
        
        log = {
            "sw_framework": "tensorflow-" + tf.__version__,
            "devices": str(tf.config.list_physical_devices()),
            "compute_precision": "",
            "batch_size_training": self.batch_size,
            "num_iterations_training": samples_training,
            "batch_size_eval": self.batch_size,
            "num_iterations_eval": len(self.x_test),
            "sample_input_shape": self.input_shape,
            }
        
        return log



class MLPMixerLayer(layers.Layer):
    """
    ## The MLP-Mixer model

    The MLP-Mixer is an architecture based exclusively on
    multi-layer perceptrons (MLPs), that contains two types of MLP layers:

    1. One applied independently to image patches, which mixes the per-location features.
    2. The other applied across patches (along channels), which mixes spatial information.

    This is similar to a [depthwise separable convolution based model](https://arxiv.org/pdf/1610.02357.pdf)
    such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization
    instead of batch normalization.
    """

    """
    ### Implement the MLP-Mixer module
    """
    
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

class Patches(layers.Layer):
    """
    ## Implement patch extraction as a layer
    """
    
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

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
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches