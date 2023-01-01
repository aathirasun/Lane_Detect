import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gr ,(5,5),0)
    canny=cv2.Canny(blur, 50,150)
    return canny
    
img=cv2.imread("C:\\Users\\aathi\\OneDrive\\Desktop\\Stuff\\school\\python programs\\rev\\test_image.jpg")
lane_img=np.copy(img)
canny =canny(lane_img)

plt.imshow(canny)
plt.show()