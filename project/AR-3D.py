from cv2 import cv2
import numpy as np
import math

'''
ORB - Oriented FAST and Rotated BRIEF
'''
class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

#homography
def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

# render 3d models
def render(img, obj, projection, model, color = False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5
    h, w, c = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        #if color is False:
        cv2.fillConvexPoly(img, imgpts, (35, 240, 90))
        #else:
        #    color = hex_to_rgb(face[-1])
        #    color = color[::-1] # reverse
        #    cv2.fillConvexPoly(img, imgpts, color)

    return img

#convert color
def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def main():
    homography = None
     # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # create ORB - Oriented FAST and Rotated BRIEF - keypoint detector
    orb = cv2.ORB_create(nfeatures=1000) # retain max 1000 features

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # image target
    model = cv2.imread('target_img.jpg')

    # calculate key point and description
    kp_model, des_model = orb.detectAndCompute(model,None) # kp: key point, des: description

    # obj file
    obj = OBJ('wolf.obj',swapyz = True)

    # Webcam
    webcam = cv2.VideoCapture(0)

    while True:
        success , imgwebcam = webcam.read()
        # find and draw the keypoints of the frame
        kp_webcam, des_webcam = orb.detectAndCompute(imgwebcam,None)

        # finding match between 2 img
        matches = bf.knnMatch(des_model, des_webcam, k=2)
        # Taking good keypoints
        good=[]
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        # compute Homography if enough matches are found
        if len(good) > 15:
            # differenciate between source points and destination points
            srcpts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dstpts = np.float32([kp_webcam[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # compute Homography
            homography, mask = cv2.findHomography(srcpts, dstpts, cv2.RANSAC, 5)

            #find boundary around model
            h, w, channel = model.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts,homography)
            # connect them with lines
            #imgwebcam = cv2.polylines(imgwebcam,[np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # if a valid homography matrix was found render object on model plane
            if homography is not None:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # render object
                    imgwebcam = render(imgwebcam, obj, projection, model)
                    #imgwebcam = render(imgwebcam, model, projection)

        cv2.imshow('result',imgwebcam)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    webcam.release()
    cv2.destroyAllWindows()
    return 0

main()