from yolox.exp.yolox_base import Exp
import torch
import cv2
from yolox.data.data_augment import ValTransform

model_path = 'my_model/best_ckpt.pth'
myexp = Exp()
model = myexp.get_model().cuda()
model.eval()
ckpt = torch.load(model_path,map_location='cpu')
model.load_state_dict(ckpt["model"])

preimg = ValTransform()
img = cv2.imread('5.jpg')
img, _ = preimg(img,None,(640,640))

img = torch.from_numpy(img).unsqueeze(0)
img = img.float().cuda()

outputs = model(img)

print(outputs.shape[0])


