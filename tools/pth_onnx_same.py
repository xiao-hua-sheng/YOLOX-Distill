import numpy
import torch
import cv2
import numpy as np
import onnx
import onnxruntime

import sys
sys.path.append(r'/home/zhangjian/yolox_distill')
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
import onnxruntime

test_size = (640, 640)
img = cv2.imread("18.jpg")
ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
preproc = ValTransform(legacy=False)
img, _ = preproc(img, None, test_size)
img = torch.from_numpy(img).unsqueeze(0)
img = img.float()

# img = np.array(img)
# sess = onnxruntime.InferenceSession('YOLOX_outputs/yolox_voc_s/coco_N_250/yolox_s_N250.onnx')
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name
# pre_onnx = sess.run([output_name],{input_name:img})
# pre_onnx = np.array(pre_onnx).reshape((8400,10))

# onnx_model = onnx.load('YOLOX_outputs/yolox_voc_s/coco_N_250/yolox_s.onnx')
# onnx.checker.check_model(onnx_model)
# output = onnx_model.graph.output
# print(output)


exp = get_exp("/home/liang/YOLOX/exps/example/yolox_voc/yolox_voc_s.py", "yolox-s")
model = exp.get_model()
model.eval()
ckpt = torch.load("YOLOX_outputs/yolox_voc_s/coco_N_250/best_ckpt.pth",map_location="cpu")
model.load_state_dict(ckpt["model"])

print(model.state_dict()['head.stems.2.conv.weight'])
# for name in model.state_dict():
#     print(name)
# pt_output = model(img)
# # print(pt_output)
# pt_output_array = pt_output.data.cpu().numpy()
# pt_output_array = pt_output_array.reshape((8400,10))
# pt_output_array = np.array(pt_output_array)
# file = open("18jpg_pt_output.txt",'w')
#
# for i,line in enumerate(pt_output_array):
#     tmp = []
#     file.write("channel %d:"%i)
#     file.write('\r\n')
#     file.write("         ")
#     for le in line:
#         tmp.append(round(le,4))
#     file.write(str(tmp))
#     file.write('\r\n')
# file.close()

