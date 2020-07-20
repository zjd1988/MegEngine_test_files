import cv2
import numpy as np

cat = cv2.imread('./cat.jpg')
cat = cv2.resize(cat,(224,224))
cat = cat[np.newaxis]  # 将cat的shape从(224,224,3) 变成 (1, 224, 224, 3)
cat = np.transpose(cat, (0, 3, 1, 2)) # nhwc -> nchw

np.save('cat.npy', np.float32(cat))