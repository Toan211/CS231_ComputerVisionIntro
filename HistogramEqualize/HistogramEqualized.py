from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# BUOC 0: Read image
img_ = cv2.imread('HistogramEqualize/DeepBlueB.jpg')

hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
h, s, img = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

L = 256
plt.hist(img.ravel(),256,[0,256])

# BUOC 1: Calculate histogram
px = np.zeros(256)


# flatten image to array
img_arr = img.reshape(-1)
for v in img_arr:
    px[v] += 1

# BUOC 2: Calculate CDF 

cdf = 0
cdf_min = np.min(px[px>0])
print('cdf_min: ', cdf_min)
npixel = len(img_arr)
out_img = img.copy()
# BUOC 3: Map intensitive value from original image to equalized one

map_val = {}
for v in range(0,L):
    # Calculate CDF(v
    cdf += px[v]
    # Calculate h(v)
    h_v = np.round((cdf - cdf_min)/(npixel - cdf_min)*(L-1))
    map_val[v] = h_v
    # DONT DO THAT: out_img[out_img==v] = h_v
    out_img[img==v] = h_v


his_img = cv2.merge((h,s,out_img))
histogram_img = cv2.cvtColor(his_img,cv2.COLOR_HSV2BGR)
plt.hist(out_img.ravel(),256,[0,256])
plt.show()

cv2.imshow('Original image', img_)
cv2.imshow('Equalized image', histogram_img)
cv2.imwrite('HistogramEqualize/EqualizedIMG.jpg', histogram_img)
cv2.waitKey()
cv2.destroyAllWindows()