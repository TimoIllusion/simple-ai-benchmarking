import tensorflow as tf


class SimpleClassificationCNN(tf.keras.Model):

    def __init__(self, num_classes, input_shape):

        super(SimpleClassificationCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=input_shape
        )
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.maxpool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes, activation="softmax")

        in_shape_with_dyn_batch = [None] + list(input_shape)
        self.build(in_shape_with_dyn_batch)

    def call(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return self.fc2(x)

