import torch
import os
import tensorrt as trt
from RIFE import Model as RifeModel  # Adjust this import if needed

WORKING_DIR = "C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation"
ENGINE_FILE_PATH = os.path.join(WORKING_DIR, 'rife_model_trt.engine')
ONNX_MODEL_PATH = os.path.join(WORKING_DIR, 'rife_model.onnx')

def export_model_to_onnx(output_path=ONNX_MODEL_PATH, img_size=(480, 640)):
    # Initialize your RIFE model and load trained weights
    model = RifeModel()
    model.load_model(path="C:\\Users\\raman\\PycharmProjects\\frame_generation\\frame_generation\\experimental_save_model", rank=0)
    model.eval()

    # Dummy input for tracing (Batch, Channels, Height, Width)
    dummy_input = torch.randn(32, 6, img_size[0], img_size[1]).cuda()

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )

    print(f"ONNX model exported to: {output_path}")



if __name__ == "__main__":
    # Export the PyTorch model to ONNX
    export_model_to_onnx(output_path=ONNX_MODEL_PATH)


