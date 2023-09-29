import tensorflow as tf
import numpy as np
from simple_ai_benchmarking.definitions import NumericalPrecision

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.log import BenchmarkResult

class TensorFlowKerasWorkload(AIWorkloadBase):

    def setup(self):
        
        if self.data_type == NumericalPrecision.MIXED_FP16_FP32:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        elif self.data_type == NumericalPrecision.DEFAULT_FP32:
            tf.keras.mixed_precision.set_global_policy('float32')
        else:
            raise NotImplementedError(f"Data type not implemented: {self.data_type}")
        
        self.model.compile(
            optimizer='adam',   
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        self.model.summary()
        
        mem_usage_gb = TensorFlowKerasWorkload.get_model_memory_usage(self.batch_size, self.model)
        print("Memory usage in GB: ", mem_usage_gb)
        
        output_shape = list(self.model.layers[-1].output_shape)
        output_shape[0] = self.num_batches * self.batch_size
        
        input_shape = list(self.model.input_shape)
        input_shape[0] = self.num_batches * self.batch_size
        
        self.input_shape_without_batch_dim = input_shape[1:]
        
        # always generate dataset on system RAM, that is why CPU is forced here
        with tf.device("/cpu:0"):
            dataset_shape = input_shape
            targets_shape = output_shape
            
            self.inputs = tf.random.normal(dataset_shape, dtype=tf.float32)
            self.targets = tf.random.uniform(targets_shape, minval=0, maxval=2, dtype=tf.int32)
            
            print("inputs shape:", self.inputs.shape)
            print("targets shape:", self.targets.shape)
            
            self.syn_dataset = tf.data.Dataset.from_tensor_slices((self.inputs, self.targets))

            self.syn_dataset = self.syn_dataset.shuffle(buffer_size=10000)
            self.syn_dataset = self.syn_dataset.batch(self.batch_size)
            self.syn_dataset = self.syn_dataset.prefetch(tf.data.AUTOTUNE)
            
    
    def train(self):
        _ = self.model.fit(self.syn_dataset, epochs=self.epochs, validation_data=None, verbose=1)
    
    def eval(self):
        raise NotImplementedError("Evaluation not implemented for TensorFlow Keras Workload")
        
    def predict(self):
        self.model.predict(self.syn_dataset, verbose=1)
  
    def build_result_log(self) -> BenchmarkResult:
        
        benchmark_result = BenchmarkResult(
            self.__class__.__name__,
            "tensorflow-" + tf.__version__,
            str(tf.config.list_physical_devices()),
            self.data_type.name,
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
    
    @staticmethod
    def get_model_memory_usage(batch_size, model):
        # credits to https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
        
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += TensorFlowKerasWorkload.get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

        number_size = 4.0
        if tf.keras.backend.floatx() == 'float16':
            number_size = 2.0
        if tf.keras.backend.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes

   
    
    
