# Modified version of Yixing Fu's "Image classification via fine-tuning with EfficientNet" from https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py (Apache License 2.0)
# Changes:
# - remove TPU and distribution strategy related code
# - remove transfer learning related code (only training from scratch is used)
# - code for visualization with matplotlib is removed
# - wrap everything into a class with setup, train and eval functionalities
# - add evaluation/inference code
# - move imports to one place
# - adjust hyperparameters
# - only take a partial amount of sample for training form dataset
# - change "stanford_dogs" dataset to "mnist"


"""
Title: Image classification via fine-tuning with EfficientNet
Author: [Yixing Fu](https://github.com/yixingfu)
Date created: 2020/06/30
Last modified: 2020/07/16
Description: Use EfficientNet with weights pre-trained on imagenet for Stanford Dogs classification.
Accelerator: TPU
"""
"""

## Introduction: what is EfficientNet

EfficientNet, first introduced in [Tan and Le, 2019](https://arxiv.org/abs/1905.11946)
is among the most efficient models (i.e. requiring least FLOPS for inference)
that reaches State-of-the-Art accuracy on both
imagenet and common image classification transfer learning tasks.

The smallest base model is similar to [MnasNet](https://arxiv.org/abs/1807.11626), which
reached near-SOTA with a significantly smaller model. By introducing a heuristic way to
scale the model, EfficientNet provides a family of models (B0 to B7) that represents a
good combination of efficiency and accuracy on a variety of scales. Such a scaling
heuristics (compound-scaling, details see
[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)) allows the
efficiency-oriented base model (B0) to surpass models at every scale, while avoiding
extensive grid-search of hyperparameters.

A summary of the latest updates on the model is available at
[here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), where various
augmentation schemes and semi-supervised learning approaches are applied to further
improve the imagenet performance of the models. These extensions of the model can be used
by updating weights without changing model architecture.

## B0 to B7 variants of EfficientNet

*(This section provides some details on "compound scaling", and can be skipped
if you're only interested in using the models)*

Based on the [original paper](https://arxiv.org/abs/1905.11946) people may have the
impression that EfficientNet is a continuous family of models created by arbitrarily
choosing scaling factor in as Eq.(3) of the paper.  However, choice of resolution,
depth and width are also restricted by many factors:

- Resolution: Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries
of some layers which wastes computational resources. This especially applies to smaller
variants of the model, hence the input resolution for B0 and B1 are chosen as 224 and
240.

- Depth and width: The building blocks of EfficientNet demands channel size to be
multiples of 8.

- Resource limit: Memory limitation may bottleneck resolution when depth
and width can still increase. In such a situation, increasing depth and/or
width but keep resolution can still improve performance.

As a result, the depth, width and resolution of each variant of the EfficientNet models
are hand-picked and proven to produce good results, though they may be significantly
off from the compound scaling formula.
Therefore, the keras implementation (detailed below) only provide these 8 models, B0 to B7,
instead of allowing arbitray choice of width / depth / resolution parameters.

## Keras implementation of EfficientNet

An implementation of EfficientNet B0 to B7 has been shipped with tf.keras since TF2.3. To
use EfficientNetB0 for classifying 1000 classes of images from imagenet, run:

```python
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```

This model takes input images of shape (224, 224, 3), and the input data should range
[0, 255]. Normalization is included as part of the model.

Because training EfficientNet on ImageNet takes a tremendous amount of resources and
several techniques that are not a part of the model architecture itself. Hence the Keras
implementation by default loads pre-trained weights obtained via training with
[AutoAugment](https://arxiv.org/abs/1805.09501).

For B0 to B7 base models, the input shapes are different. Here is a list of input shape
expected for each model:

| Base model | resolution|
|----------------|-----|
| EfficientNetB0 | 224 |
| EfficientNetB1 | 240 |
| EfficientNetB2 | 260 |
| EfficientNetB3 | 300 |
| EfficientNetB4 | 380 |
| EfficientNetB5 | 456 |
| EfficientNetB6 | 528 |
| EfficientNetB7 | 600 |

When the model is intended for transfer learning, the Keras implementation
provides a option to remove the top layers:
```
model = EfficientNetB0(include_top=False, weights='imagenet')
```
This option excludes the final `Dense` layer that turns 1280 features on the penultimate
layer into prediction of the 1000 ImageNet classes. Replacing the top layer with custom
layers allows using EfficientNet as a feature extractor in a transfer learning workflow.

Another argument in the model constructor worth noticing is `drop_connect_rate` which controls
the dropout rate responsible for [stochastic depth](https://arxiv.org/abs/1603.09382).
This parameter serves as a toggle for extra regularization in finetuning, but does not
affect loaded weights. For example, when stronger regularization is desired, try:

```python
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
```
The default value is 0.2.

## Example: EfficientNetB0 for Stanford Dogs.

EfficientNet is capable of a wide range of image classification tasks.
This makes it a good model for transfer learning.
As an end-to-end example, we will show using pre-trained EfficientNetB0 on
[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) dataset.

"""

"""

## Setup and data loading

This example requires TensorFlow 2.3 or above.

To use TPU, the TPU runtime must match current running TensorFlow
version. If there is a mismatch, try:

```python
from cloud_tpu_client import Client
c = Client()
c.configure_tpu_version(tf.__version__, restart_type="always")
```
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from simple_ai_benchmarking import AIWorkload

# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
#     print("Device:", tpu.master())
#     strategy = tf.distribute.TPUStrategy(tpu)
# except ValueError:
#     print("Not connected to a TPU runtime. Using CPU/GPU strategy")
#     strategy = tf.distribute.MirroredStrategy()


"""
### Loading data

Here we load data from [tensorflow_datasets](https://www.tensorflow.org/datasets)
(hereafter TFDS).
Stanford Dogs dataset is provided in
TFDS as [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs).
It features 20,580 images that belong to 120 classes of dog breeds
(12,000 for training and 8,580 for testing).

By simply changing `dataset_name` below, you may also try this notebook for
other datasets in TFDS such as
[cifar10](https://www.tensorflow.org/datasets/catalog/cifar10),
[cifar100](https://www.tensorflow.org/datasets/catalog/cifar100),
[food101](https://www.tensorflow.org/datasets/catalog/food101),
etc. When the images are much smaller than the size of EfficientNet input,
we can simply upsample the input images. It has been shown in
[Tan and Le, 2019](https://arxiv.org/abs/1905.11946) that transfer learning
result is better for increased resolution even if input images remain small.

For TPU: if using TFDS datasets,
a [GCS bucket](https://cloud.google.com/storage/docs/key-terms#buckets)
location is required to save the datasets. For example:

```python
tfds.load(dataset_name, data_dir="gs://example-bucket/datapath")
```

Also, both the current environment and the TPU service account have
proper [access](https://cloud.google.com/tpu/docs/storage-buckets#authorize_the_service_account)
to the bucket. Alternatively, for small datasets you may try loading data
into the memory and use `tf.data.Dataset.from_tensor_slices()`.
"""




class EfficientNet(AIWorkload):
    """
    The [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.
    """
    
    def __init__(self):
        
        # IMG_SIZE is determined by EfficientNet model choice
        self.IMG_SIZE = 224


        self.batch_size = 64
        self.epochs = 3  # @param {type: "slider", min:10, max:100}
        self.num_training_batches = 10
        self.num_inference_batches = 10

        dataset_name = "mnist" #"stanford_dogs"
        (self.ds_train, self.ds_test), ds_info = tfds.load(
            dataset_name, split=["train", "test"], with_info=True, as_supervised=True
        )
        self.NUM_CLASSES = ds_info.features["label"].num_classes


      

    def setup(self):
        """
        When the dataset include images with various size, we need to resize them into a
        shared size. The Stanford Dogs dataset includes only images at least 200x200
        pixels in size. Here we resize the images to the input size needed for EfficientNet.
        """

        self.size = (self.IMG_SIZE, self.IMG_SIZE)
        self.ds_train = self.ds_train.map(lambda image, label: (tf.image.resize(image, self.size), label))
        self.ds_test = self.ds_test.map(lambda image, label: (tf.image.resize(image, self.size), label))



        """
        ### Data augmentation

        We can use the preprocessing layers APIs for image augmentation.
        """


        img_augmentation = Sequential(
            [
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomFlip(),
                layers.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )


        """
        ### Prepare inputs

        Once we verify the input data and augmentation are working correctly,
        we prepare dataset for training. The input data are resized to uniform
        `IMG_SIZE`. The labels are put into one-hot
        (a.k.a. categorical) encoding. The dataset is batched.

        Note: `prefetch` and `AUTOTUNE` may in some situation improve
        performance, but depends on environment and the specific dataset used.
        See this [guide](https://www.tensorflow.org/guide/data_performance)
        for more information on data pipeline performance.
        """

        # One-hot / categorical encoding
        def input_preprocess(image, label):
            label = tf.one_hot(label, self.NUM_CLASSES)
            return image, label

        self.ds_train = self.ds_train.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.batch(batch_size=self.batch_size, drop_remainder=True)
        self.ds_train = self.ds_train.prefetch(tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.take(self.num_training_batches)

        self.ds_test = self.ds_test.map(input_preprocess)
        self.ds_test = self.ds_test.batch(batch_size=self.batch_size, drop_remainder=True)
        self.ds_test = self.ds_test.take(self.num_inference_batches)


        """
        ## Training a model from scratch

        We build an EfficientNetB0 with 120 output classes, that is initialized from scratch:

        Note: the accuracy will increase very slowly and may overfit.
        """

        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        inputs = layers.Input(shape=self.input_shape)
        x = img_augmentation(inputs)
        outputs = EfficientNetB0(include_top=True, weights=None, classes=self.NUM_CLASSES)(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model.summary()    
    

    def train(self):
        

        _ = self.model.fit(self.ds_train, epochs=self.epochs, validation_data=self.ds_test, verbose=1)

        """
        Training the model is relatively fast (takes only 20 seconds per epoch on TPUv2 that is
        available on Colab). This might make it sounds easy to simply train EfficientNet on any
        dataset wanted from scratch. However, training EfficientNet on smaller datasets,
        especially those with lower resolution like CIFAR-100, faces the significant challenge of
        overfitting.

        Hence training from scratch requires very careful choice of hyperparameters and is
        difficult to find suitable regularization. It would also be much more demanding in resources.
        Plotting the training and validation accuracy
        makes it clear that validation accuracy stagnates at a low value.
        """
        
    def eval(self):
        loss, accuracy = self.model.evaluate(self.ds_test)
        print(f"Test loss: {round(loss, 2)}")
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")    
    
    def build_log_dict(self):
        
        log = {
            "sw_framework": "tensorflow-" + tf.__version__,
            "devices": str(tf.config.list_physical_devices()),
            "compute_precision": "",
            "batch_size_training": self.batch_size,
            "num_iterations_training": self.num_training_batches * self.batch_size * self.epochs, #TODO: check if number is correct, due to "drop_reminder"
            "batch_size_eval": self.batch_size,
            "num_iterations_eval": self.num_inference_batches * self.batch_size, #TODO: check if number is correct, due to "drop_reminder"
            "sample_input_shape": self.input_shape,
            }
        
        return log    
    