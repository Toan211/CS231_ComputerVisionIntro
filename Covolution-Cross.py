from cv2 import cv2
import numpy as np

import matplotlib.pyplot as plt

#Cross-Correlation
def cross(img,fil,width,height):
    a = int((len(fil[0]) -1)/2)
    b = len(fil[0])
    result = []
    for x in range(0+a,height-a):
        for y in range(0+a,width-a):
            tmp = 0
            z = 0
            for h in range(0,b):
                z = img[x-1+h]
                for w in range(0,b):
                    tmp += z[y-1+w] * fil[h][w]
            result.append(tmp)

    return np.array(result, dtype=int).reshape(([height-a*2,width-a*2]))

#Convolution
def convo(img,fil,width,height):
    a = int((len(fil[0]) - 1) / 2)
    b = len(fil[0])
    result = []
    for x in range(0 + a, height - a):
        for y in range(0 + a, width - a):
            tmp = 0
            z = 0
            for h in range(0, b):
                z = img[x - 1 + h]
                for w in range(0, b):
                    tmp += z[y - 1 + w] * fil[b-h-1][b-w-1]
            result.append(tmp)

    return np.array(result, dtype=int).reshape(([height - a * 2,width - a * 2]))

#Load image

img = np.array(cv2.imread("DeepBlueB.jpg", 0))
n,m = img.shape

filterr =np.array([(-1,0,1),
			        (-1,0,1),
			        (-1,0,1)])/9.0

r1 = convo(img,filterr,m,n)
r2 = cross(img,filterr,m,n)


plt.imshow(r2, cmap= 'gray')
plt.title("Convo", size=30)

plt.show()

'''
plt.imshow(img, cmap= 'gray',)
plt.axis("off")
plt.show()
'''