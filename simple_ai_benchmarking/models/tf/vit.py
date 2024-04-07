import tensorflow as tf
import tensorflow_hub as hub

#TODO: fix num_classes and model
class VisionTransformer(tf.keras.Model):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        
        model_url = 'https://tfhub.dev/sayakpaul/vit_b16_classification/1'
        self.model = hub.KerasLayer(model_url, trainable=True)

    def call(self, inputs):
        return self.model(inputs)
