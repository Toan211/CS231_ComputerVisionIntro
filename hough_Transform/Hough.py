'''
HoughLine function of cv2 library

# The below for loop runs till r and theta values  
# are in the range of the 2d array 
p = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 150  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
lines = cv2.HoughLinesP(edges, p, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
for line in lines:
    for x1,y1,x2,y2 in line:
    	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)
'''
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def focus_zone(img, focus_zone):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, focus_zone, match_mask_color) # Fill black to region out of focus_zone
    masked_image = cv2.bitwise_and(img, mask) # Keep pixel where isnt black in mask
    return masked_image

def line_detection_non_vectorized(image, num_thetas=180, t_count=130):

	# Load gray image and detect edge
	edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edge_image = cv2.GaussianBlur(edge_image, (5,5), 1) 
	edge_image = cv2.Canny(edge_image,100,150,apertureSize = 3)
	edge_height, edge_width = edge_image.shape[:2]

	d = int(np.sqrt(np.square(edge_height) + np.square(edge_width)))
	dtheta = 180 / num_thetas
	thetas = np.arange(0, 180, step=dtheta)

	cos_thetas = np.cos(thetas)
	sin_thetas = np.sin(thetas)
	hough_matrix = np.zeros((d, 180))

	# Calculate the HoughLine
	for y in range(edge_height):
		for x in range(edge_width):
			if edge_image[y][x] != 0:
				for theta_idx in range(len(thetas)):
					p = int((x * cos_thetas[theta_idx]) + (y * sin_thetas[theta_idx]))
					theta = thetas[theta_idx]
					hough_matrix[p][theta_idx] += 1

	# Choose and draw line
	for y in range (hough_matrix.shape[0]):
		for x in range (hough_matrix.shape[1]):
			if hough_matrix[y][x] > t_count:
				p = y
				theta = thetas[x]
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = (a * p)
				y0 = (b * p)
				x1 = int(x0 + 1000 * (-b))
				y1 = int(y0 + 1000 * (a))
				x2 = int(x0 - 1000 * (-b))
				y2 = int(y0 - 1000 * (a))
				cv2.line(image,(x1,y1),(x2,y2),(0,255,0),10)

	return image

def main():
	image = cv2.imread('test.jpg')

	height, width = image.shape[:2]
	zone = [(0,height),(0,230),(300,100),(640,240),(width,height)] # Only work with this image

	focused_img = focus_zone(image,np.array([zone], np.int32))
	focused_img[100:,:] = line_detection_non_vectorized(focused_img[100:,:])

	result = cv2.bitwise_or(image,focused_img) # Take all pixel where isnt black between 2 imgs 

	plt.imshow(result)		
	plt.show()

if __name__ == '__main__':
	main()