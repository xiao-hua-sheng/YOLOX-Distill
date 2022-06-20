import cv2
import numpy as np
import onnxruntime as ort

img = cv2.imread('imgs/5.jpg')
input_blob = cv2.dnn.blobFromImage(img,1/255,(640,640))

# ort_session = ort.InferenceSession('my_model/out.onnx')
ort_session = ort.InferenceSession('my_model/2021_12_17_model2-sim.onnx')
onnx_input_name = ort_session.get_inputs()[0].name
onnx_outputs_name = ort_session.get_outputs()
output_names = []
for o in onnx_outputs_name:
    output_names.append(o.name)

onnx_result = ort_session.run(output_names,input_feed={onnx_input_name:input_blob})
onnx_result = onnx_result[0]
print(onnx_result[0][6400])