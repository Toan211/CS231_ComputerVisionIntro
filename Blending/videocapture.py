from cv2 import cv2
import numpy as np

img = cv2.imread('Blending/girl.jpg',0)
height, width = img.shape
size = (width,height)
cap = cv2.VideoCapture('Blending/Fire.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening")
img1_array = []
i = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    b = cv2.resize(frame,(width,height),fx=0,fy=0, interpolation = cv2.INTER_AREA)
    img1_array.append(b)
    i = i + 1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('Q'):
      break
  # Break the loop
  else: 
    break

for i in range(0,420):
    foreground = img1_array[i]
    background = cv2.imread("Blending/girl.jpg")
    alpha = cv2.imread("Blending/mask.png")
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    # Display image
    cv2.imshow("outImg", outImage/255)
    print(i)
    if cv2.waitKey(25) & 0xFF == ord('Q'):
      break
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
