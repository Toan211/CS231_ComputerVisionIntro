from cv2 import cv2
import numpy as np

def energy(img):
    Gradient = cv2.Canny(img,50,100)
    a,b = Gradient.shape
    print(a,b)
    return Gradient
def min_seam(img):
    r,c = img.shape
    energy1 = energy(img)
    Temp = np.copy(energy1)
    Backtrack = np.zeros_like(energy1)
    for i in range (1, r):
        for j in range(0, c):
            if j == 0:
                index = np.argmin(Temp[i-1,j:j+2])
                Backtrack[i,j] = index
                min_energy = Temp[i-1,index]
            elif j == c:
                index = np.argmin(Temp[i-1,j-1:j+1])
                Backtrack[i,j] = index + j - 1
                min_energy = Temp[i-1,index + j - 1]
            else:
                index = np.argmin(Temp[i-1,j-1:j+2])
                Backtrack[i,j] = index + j - 1
                min_energy = Temp[i-1,index+j-1]
            Temp[i,j] = Temp[i,j] + min_energy
    return Temp, Backtrack
def delete_column(img):
    r, c = img.shape
    Temp, Backtrack = min_seam(img)
    Del = np.ones_like(img, dtype=np.bool)
    j = np.argmin(Temp[-1])
    for i in reversed(range(r)):
        Del[i,j] = False
        j = Backtrack[i,j]
    img = img[Del].reshape(r, c-1)
    return img

def crop_c(img, scale_c):
    r, c = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c):
        img = delete_column(img)

    return img
img2 = cv2.imread('seam_carving/blue.jpg')
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
out = crop_c(gray, 0.75)
cv2.imwrite('SeamCarving.jpg', out)

cv2.waitKey(0)