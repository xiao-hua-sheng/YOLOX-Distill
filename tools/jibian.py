import cv2
import numpy as np
img=cv2.imread('1.jpg')
rows,cols=img.shape[:2]
print(rows,cols)
pts1 = np.float32([[27,53],[105,1000],[500,101],[500,1000]])
pts2 = np.float32([[5,5],[5,rows-5],[cols-5,5],[cols-5,rows-5]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(cols,rows))
cv2.imwrite("2.jpg",dst)

