import os
import logging
import tensorrt as trt

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage.
    """

    def __init__(self, verbose=False, workspace=16):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in GB. Increased for maximum optimization.
        """
        self.trt_logger = trt.Logger(trt.Logger.ERROR)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()

        # Increase the workspace memory pool size for maximum performance
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(1)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                return False

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        assert self.batch_size > 0
        return True

    def create_engine(self, engine_path, precision="fp16", use_int8=False):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16'.
        :param use_int8: Enable INT8 precision mode if hardware supports it.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        # Set precision flags
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)

        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        # Build and serialize the engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            return False

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)
        return True


def main():
    # Define paths and parameters directly
    onnx_model_path = "C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\rife_model.onnx"  # Change to your ONNX model path
    engine_file_path = "path_to_save_model.trt"  # Change to your desired output path
    precision_mode = "fp16"  # Choose "fp16" or "fp32"
    use_int8_precision = False  # Set to True if you want to use INT8 (ensure your hardware supports it)
    workspace_size_gb = 10  # Max memory workspace in GB for optimization
    verbose_logging = True  # Enable verbose logging for detailed output

    # Create an EngineBuilder instance and build the model
    builder = EngineBuilder(verbose=verbose_logging, workspace=workspace_size_gb)

    # Load the ONNX model and create the TensorRT engine
    if builder.create_network(onnx_model_path):
        builder.create_engine(engine_file_path, precision=precision_mode, use_int8=use_int8_precision)


if __name__ == "__main__":
    main()
