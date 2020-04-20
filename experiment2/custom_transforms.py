import cv2
import numpy as np

class BgrToOpp(object):
    
    def __call__(self, img, target, normalize=True):
#         img = np.asanyarray(img)
        if normalize:
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        (B, G, R) = cv2.split(img.astype("float"))
        # compute rg = R - G
        O1 = ((R + G + B) -1.5)/ 1.5
        O2 = ((R - G))
        O3 = ((R + G) - (2 * B))/2
        img = cv2.merge((O1,O2,O3))
        img = img.astype(np.float32)
        return img, target