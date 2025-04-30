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

    def __init__(self, verbose=False, workspace=8):
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
            trt.MemoryPoolType.WORKSPACE,workspace * (2 ** 30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path,input_shape):
        """
            Parse the ONNX graph and create the corresponding TensorRT network definition.

            Parameters
            ----------
            onnx_path: String contain file path
                The path to the ONNX graph to load.

            input_shape: List of tuple, each having len==3
                Each tuple has 3 values, i.e min,opt,max
           """

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(explicit_batch)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                return False

        input_tensor = self.network.get_input(0)
        if not input_tensor:
            log.error("No input tensor found in the ONNX model!")
            return False

        log.info(f"Found input tensor: {input_tensor.name}")

        if len(input_tensor[0]) !=3:
            raise ValueError("The input shape does not contain either of these values: min,opt,max")

        for i in input_shape:
            profile = self.builder.create_optimization_profile()
            profile.set_shape(input_tensor.name,
                              min = i[0],
                              opt = i[1],
                              max = i[2])
            self.config.add_optimization_profile(profile)

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

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)

        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            return False

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)
        return True