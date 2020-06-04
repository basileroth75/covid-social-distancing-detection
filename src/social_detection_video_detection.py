# import the necessary packages
#import adrian_detection
import numpy as np
import imutils
import cv2
import os
import math
import itertools
import yaml

""" Load the config parameters from the YAML file
"""
with open("../conf/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width, height = 0
corner_points = []
print("[ Loading config file ] ...")
for section in cfg:
	corner_points.append(cfg["image_coordinates"]["p1"])
	corner_points.append(cfg["image_coordinates"]["p2"])
	corner_points.append(cfg["image_coordinates"]["p3"])
	corner_points.append(cfg["image_coordinates"]["p4"])
	width = cfg["image_coordinates"]["width"]
	height = cfg["image_coordinates"]["height"]



""" Load the YOLO weights and the config parameter
"""
print("[ Loading YOLO model ] ...")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Compute the transformation matrix from the original frame
matrix = compute_perspective_transform(corner_points,width,height)
# Apply the transformation to the image to get the format of transformed plan
imgOG = cv2.warpPerspective(image,matrix,(width,height))
height,width,_ = imgOG.shape


######################################################
#########									 #########
# 				START THE VIDEO STREAM               #
#########									 #########
######################################################
vs = cv2.VideoCapture("../video/pedestrians.mp4")
# Loop until the end of the video stream
while True:

	# Create a full black frame 
	blank_image = np.zeros((height,width,3), np.uint8)
	
	# Load the frame and test if it has reache the end of the video
	(frame_exists, frame) = vs.read()
	if not frame_exists:
		break
	else:
		# Detect the person in the frame and test if there is more 
		results = detect_people(frame, net, ln, 0)

		# Test if there is more than 2 people in the frame or not 
		if len(results) >= 2:
			list_downoids = list()
			# loop over the results
			for (i, (prob, bbox, centroid)) in enumerate(results):
				# extract the bounding box and centroid coordinates, then
				# initialize the color of the annotation
				(startX, startY, endX, endY) = bbox
				(cX, cY) = centroid
				color = (0, 255, 0)
				dist_x = int(math.sqrt((startX - endX)**2)/2)
				dist_y = int(math.sqrt((startY - endY)**2)/2)

				#cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				# cv2.circle(frame, (cX, cY+dist_y), 5, color, 1)
				list_downoids.append([cX, cY+dist_y])


			list_indexes = list(itertools.combinations(range(len(list_downoids)), 2))
			for i,pair in enumerate(itertools.combinations(list_downoids, r=2)):
				# print("----")
				# print(pair)
				# print(math.sqrt((pair[0][0]-pair[1][0])**2+(pair[0][1]-pair[1][1])**2))
				if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < 100:
					index_pt1 = list_indexes[i][0]
					index_pt2 = list_indexes[i][1]
					(startX, startY, endX, endY) = results[index_pt1][1]
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
					(startX, startY, endX, endY) = results[index_pt2][1]
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
					(x1, y1) = results[index_pt1][2]
					(x2, y2) = results[index_pt2][2]



	cv2.imshow("Bird view", blank_image)
	cv2.imwrite("blank_image.jpg", blank_image)
	cv2.imshow("Frame", frame)
	cv2.imwrite("Frame.jpg", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	#break