from abc import abstractmethod, ABC

import tensorflow as tf

from simple_ai_benchmarking.log import BenchmarkResult

#TODO: add TensorFlowAIWorkload as subclass
class AIWorkload(ABC):
    
    def __init__(self, model, epochs: int, num_batches: int, batch_size: int, device_name: str):
        self.model = model
        self.epochs = epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device_name = device_name

    @abstractmethod
    def setup(self):
        pass    
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass    
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def build_result_log(self) -> BenchmarkResult:
        return BenchmarkResult()    


class SimpleClassificationCNN:
    
    @staticmethod
    def build_model(num_classes, input_shape):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        
        # Compile the model
        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        return model
    

class TensorFlowWorkload(AIWorkload):

    def setup(self):
        
        self.model.summary()
        
        output_shape = list(self.model.layers[-1].output_shape)
        output_shape[0] = self.num_batches * self.batch_size
        
        input_shape = list(self.model.input_shape)
        input_shape[0] = self.num_batches * self.batch_size
        
        self.input_shape_without_batch_dim = input_shape[1:]
        
        # always generate dataset on system RAM, that is why CPU is forced here
        with tf.device("/cpu:0"):
            dataset_shape = input_shape
            targets_shape = output_shape
            
            self.dataset_train = tf.random.normal(dataset_shape, dtype=tf.float32)
            # pseudo one hot
            self.targets_train = tf.random.uniform(targets_shape, minval=0, maxval=2, dtype=tf.int32) 
            
            self.dataset_val = tf.random.normal(dataset_shape, dtype=tf.float32)
             # pseudo one hot
            self.targets_val = tf.random.uniform(targets_shape, minval=0, maxval=2, dtype=tf.int32)
            
            print("dataset shape:", self.dataset_train.shape)
            print("targets shape:", self.targets_train.shape)
    
    def train(self):
        _ = self.model.fit(self.dataset_train, self.targets_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=None, verbose=1)
    
    def eval(self):
        _, _ = self.model.evaluate(self.dataset_val, self.targets_val)
        
    def predict(self):
        _ = self.model.predict(self.dataset_val)
        
    def build_result_log(self) -> BenchmarkResult:
        
        benchmark_result = BenchmarkResult(
            self.__class__.__name__,
            "tensorflow-" + tf.__version__,
            str(tf.config.list_physical_devices()),
            "",
            self.batch_size,
            self.num_batches * self.batch_size * self.epochs, #TODO: check if number is correct, due to "drop_reminder"
            self.batch_size,
            self.num_batches * self.batch_size, #TODO: check if number is correct, due to "drop_reminder"
            self.input_shape_without_batch_dim,
            None,
            None,
            None,
            None
        )
        
        return benchmark_result
        
        

    #         _ = self.model.fit(self.ds_train, epochs=self.epochs, validation_data=self.ds_test, verbose=1)

    #         """
    #         Training the model is relatively fast (takes only 20 seconds per epoch on TPUv2 that is
    #         available on Colab). This might make it sounds easy to simply train EfficientNet on any
    #         dataset wanted from scratch. However, training EfficientNet on smaller datasets,
    #         especially those with lower resolution like CIFAR-100, faces the significant challenge of
    #         overfitting.

    #         Hence training from scratch requires very careful choice of hyperparameters and is
    #         difficult to find suitable regularization. It would also be much more demanding in resources.
    #         Plotting the training and validation accuracy
    #         makes it clear that validation accuracy stagnates at a low value.
    #         """
        
    # def eval(self):
    #     with tf.device(self.device_name):
            
    #         loss, accuracy = self.model.evaluate(self.ds_test)
    #         print(f"Test loss: {round(loss, 2)}")
    #         print(f"Test accuracy: {round(accuracy * 100, 2)}%")    
    
    