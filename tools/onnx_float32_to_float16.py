import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

# Update the input name and path for your ONNX model
input_onnx_model = '../my_model/2021_11_30_model1-opt.onnx'
# Change this path to the output name and path for your float16 ONNX model
output_onnx_model = '../my_model/2021_11_30_model1-opt-f16.onnx'
# Load your model
onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# Convert tensor float type from your input ONNX model to tensor float16
onnx_model = convert_float_to_float16(onnx_model)
# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)