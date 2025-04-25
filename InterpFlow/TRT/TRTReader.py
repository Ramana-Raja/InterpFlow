import onnxruntime as ort
import numpy as np
import torch
# Load the ONNX model
import time
import pycuda.driver as cuda

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time


class TRTInference:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Gather all I/O tensor names
        self.input_names = [name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.output_names = [name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

        # Allocate buffers
        self.bindings = {}
        for name in self.input_names + self.output_names:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            size = np.prod(shape)

            host_mem = cuda.pagelocked_empty(int(size), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[name] = {
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype,
            }

    def infer(self, input_array):
        input_name = self.input_names[0]
        output_name = self.output_names[0]

        np.copyto(self.bindings[input_name]["host"], input_array.ravel())

        cuda.memcpy_htod_async(self.bindings[input_name]["device"], self.bindings[input_name]["host"], self.stream)
        self.context.set_tensor_address(input_name, int(self.bindings[input_name]["device"]))
        self.context.set_tensor_address(output_name, int(self.bindings[output_name]["device"]))
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.bindings[output_name]["host"], self.bindings[output_name]["device"], self.stream)

        self.stream.synchronize()
        return np.array(self.bindings[output_name]["host"].reshape(self.bindings[output_name]["shape"]))
