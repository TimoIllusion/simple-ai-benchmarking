from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.config_structures import (
    InferenceConfig,
    AIFramework,
    AIStage,
)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTInference(AIWorkload):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self.cfg: InferenceConfig  # for type hinting
        self.engine = None
        self.context = None
        self.inputs, self.outputs, self.bindings, self.stream = None, None, None, None

    def setup(self) -> None:
        # Load the TensorRT engine from a previously serialized file or directly from an ONNX model
        
        self.engine = self.load_engine("SimpleClassificationCNN_pt.onnx")
        
        # Create an execution context, which is required for executing the model
        self.context = self.engine.create_execution_context()

    def load_engine(self, model_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)

        # Check if a serialized engine exists at the model path
        try:
            with open(model_path, 'rb') as f:
                engine_data = f.read()
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
            print("Loaded serialized engine.")
        except FileNotFoundError:
            # If no serialized engine, convert from ONNX
            engine = self.build_engine_from_onnx(model_path, TRT_LOGGER)
            
            # Optionally, serialize the engine for future use
            with open(model_path, 'wb') as f:
                f.write(engine.serialize())
            print("Built and serialized engine from ONNX model.")

        return engine
    
    def build_engine_from_onnx(self, onnx_file_path: str, logger: trt.Logger):
        builder = trt.Builder(logger)
        
        network = builder.create_network(flags=trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB of workspace size
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        return engine

    def prepare_execution(self) -> None:
        # Allocate buffers and create a CUDA stream
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def execute(self) -> None:
        # Perform inference using TensorRT
        cuda.memcpy_htod_async(self.inputs[0], self.input_data, self.stream)
        self.context.execute_async(
            batch_size=self.cfg.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self.output_data, self.outputs[0], self.stream)
        self.stream.synchronize()

    def allocate_buffers(self):
        # Allocate memory for inputs/outputs
        pass

    def cleanup(self) -> None:
        # Release resources
        pass

    def _get_ai_framework_name(self) -> str:
        return "TensorRT"

    def _get_ai_framework_version(self) -> str:
        return trt.__version__

    def _get_ai_framework_extra_info(self) -> str:
        return "N/A"

    def _get_ai_stage(self) -> AIStage:
        return AIStage.INFERENCE
