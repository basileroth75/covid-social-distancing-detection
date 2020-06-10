from bird_view_transfo import compute_perspective_transform,compute_point_perspective_transformation
from adrian_detection import detect_people
from tf_model_object_detection import Model 
from colors import bcolors
import numpy as np
import itertools
import imutils
import time
import math
import glob
import yaml
import cv2
import os


def get_human_box_detection(boxes,scores,classes,height,width):
	""" 
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes :
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
	array_boxes = list() # Create an empty list
	for i in range(boxes.shape[1]):
		# If the class of the detected object is 1 and the confidence of the prediction is > 0.6
		if classes[i] == 1 and scores[i] > 0.60:
			# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
			# To transform the box value into pixel coordonate values.
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes


def get_points_from_box(box):
	"""
	Get the center of the bounding and the point "on the ground"
	@ box : 2 points representing the bounding box
	"""
	# Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
	center_x = int(((box[1]+box[3])/2))
	center_y = int(((box[0]+box[2])/2))
	# Coordiniate on the ground (Center X)
	center_y_ground = center_y + ((box[2] - box[0])/2)
	return (center_x,center_y),(center_x,int(center_y_ground))
""" 
Load the config parameters from the YAML file
"""
print(bcolors.WARNING +"[ Loading config file for the bird view transformation ] "+ bcolors.ENDC)
with open("../conf/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]
print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )



""" 
Load the YOLO weights and the config parameter

print("[ Loading YOLO model ] ...")
print(bcolors.WARNING +"[ Loading config file ] "+ bcolors.ENDC)
net = cv2.dnn.readNetFromDarknet("../yolo-coco/yolov3.cfg", "../yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
"""

######################################### 
#		     Select the model 			#
#########################################
model_names_list = [name for name in os.listdir("../models/.") if name.find(".") == -1]
for index,model_name in enumerate(model_names_list):
    print(" - {} [{}]".format(model_name,index))
model_num = input(" Please select the number related to the model that you want : ")
if model_num == "":
	model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 
else :
	model_path = "../models/"+model_names_list[int(model_num)]+"/frozen_inference_graph.pb"
print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
model = Model(model_path)
print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)


######################################### 
#		     Select the video 			#
#########################################
video_names_list = [name for name in os.listdir("../video/") if name.endswith(".mp4") or name.endswith(".avi")]
for index,video_name in enumerate(video_names_list):
    print(" - {} [{}]".format(video_name,index))
video_num = input("Enter the exact name of the video (including .mp4 or else) : ")
if video_num == "":
	video_path="../video/PETS2009.avi"  
else :
	video_path = "../video/"+video_names_list[int(video_num)]




# Compute the transformation matrix from the original frame
matrix = compute_perspective_transform(corner_points,width_og,height_og)
image = cv2.imread(img_path)
imgOutput = cv2.warpPerspective(image,matrix,(width_og,height_og))
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1] 
dim = (width, height)



######################################################
#########									 #########
# 				START THE VIDEO STREAM               #
#########									 #########
######################################################

distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")

vs = cv2.VideoCapture(video_path)
output_video_1,output_video_2 = None,None
# Loop until the end of the video stream
while True:
	start_time = time.time()
	# Create a full black frame 
	print("Loaded image")
	img = cv2.imread("../img/chemin_1.png")
	# resize image
	blank_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	
	# Load the frame and test if it has reache the end of the video
	(frame_exists, frame) = vs.read()
	if not frame_exists:
		break
	else:
		# Resize the image to the correct size
		frame = imutils.resize(frame, width=int(size_frame))
		
		# Make the predictions for this frame
		(boxes, scores, classes) =  model.predict(frame)

		scores = scores[0].tolist()
		classes = [int(x) for x in classes[0].tolist()]
		array_boxes_detected = get_human_box_detection(boxes,scores,classes,frame.shape[0],frame.shape[1])
		if len(array_boxes_detected) >= 2:
			array_centroids = list()
			array_groundpoints = list()
			for index,box in enumerate(array_boxes_detected):
				cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(255,0,0),1)
				centroid,ground_point = get_points_from_box(box)
				array_centroids.append(centroid)
				array_groundpoints.append(centroid)
				# cv2.circle(frame, (centroid[0],centroid[1]), 1, (255, 255, 255), 2)
				# cv2.circle(frame, (ground_point[0],ground_point[1]), 1, (255, 0, 0), 2)
				# cv2.circle(frame, (groundpoint[0],groundpoint[1]), 5, (0, 255, 0), 2)

			transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
			for point in transformed_downoids:
				x,y = point
				cv2.circle(blank_image, (x,y), 20, (0, 255, 0), 2)
				cv2.circle(blank_image, (x,y), 3, (0, 255, 0), -1)

			list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
			for i,pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
				if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
					cv2.circle(blank_image, (pair[0][0],pair[0][1]), 20, (0, 0, 255), 2)
					cv2.circle(blank_image, (pair[0][0],pair[0][1]), 3, (0, 0, 255), -1)
					cv2.circle(blank_image, (pair[1][0],pair[1][1]), 20, (0, 0, 255), 2)
					cv2.circle(blank_image, (pair[1][0],pair[1][1]), 3, (0, 0, 255), -1)
					index_pt1 = list_indexes[i][0]
					index_pt2 = list_indexes[i][1]
					#(startX, startY, endX, endY) = array_boxes_detected[index_pt1][1]
					cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),(255,0,0),1)
					# cv2.rectangle(frame, (startX, startY), (endX, endY), ( 0, 0,255), 2)
					# (startX, startY, endX, endY) = results[index_pt2][1]
					cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),(255,0,0),1)

					# cv2.rectangle(frame, (startX, startY), (endX, endY), ( 0, 0,255), 2)
					# (x1, y1) = results[index_pt1][2]
					# (x2, y2) = results[index_pt2][2]

		# print(num)
		# Detect the person in the frame and test if there is more 
		# results = detect_people(frame, net, ln, 0)

		"""
		# Test if there is more than 2 people in the frame or not 
		if len(results) >= 2:
			list_downoids = list()
				# loop over the results
				for (i, (prob, bbox, centroid)) in enumerate(results):
					# extract the bounding box and centroid coordinates, then
					# initialize the color of the annotation
					(startX, startY, endX, endY) = bbox
					(cX, cY) = centroid
					dist_x = int(math.sqrt((startX - endX)**2)/2)
					dist_y = int(math.sqrt((startY - endY)**2)/2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					# cv2.circle(frame, (cX, cY+dist_y), 5, (0, 255, 0), 1)
					list_downoids.append([cX, cY+dist_y])

				transformed_downoids = compute_point_perspective_transformation(matrix,list_downoids)
				for point in transformed_downoids:
					x,y = point
					cv2.circle(blank_image, (x,y), 20, (0, 255, 0), 2)
					cv2.circle(blank_image, (x,y), 3, (0, 255, 0), -1)


				list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
				for i,pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
					if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
						cv2.circle(blank_image, (pair[0][0],pair[0][1]), 20, (0, 0, 255), 2)
						cv2.circle(blank_image, (pair[0][0],pair[0][1]), 3, (0, 0, 255), -1)
						cv2.circle(blank_image, (pair[1][0],pair[1][1]), 20, (0, 0, 255), 2)
						cv2.circle(blank_image, (pair[1][0],pair[1][1]), 3, (0, 0, 255), -1)
						index_pt1 = list_indexes[i][0]
						index_pt2 = list_indexes[i][1]
						(startX, startY, endX, endY) = results[index_pt1][1]
						cv2.rectangle(frame, (startX, startY), (endX, endY), ( 0, 0,255), 2)
						(startX, startY, endX, endY) = results[index_pt2][1]
						cv2.rectangle(frame, (startX, startY), (endX, endY), ( 0, 0,255), 2)
						(x1, y1) = results[index_pt1][2]
						(x2, y2) = results[index_pt2][2]

		"""
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), (0, 255, 0), thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), (0, 255, 0), thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), (0, 255, 0), thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), (0, 255, 0), thickness=1)

	cv2.imshow("Bird view", blank_image)
	cv2.imshow("Original picture", frame)
	key = cv2.waitKey(1) & 0xFF

	end_time = time.time()
	print("Elapsed Time:", end_time-start_time)

	if output_video_1 is None:
		fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
	elif output_video_1 is not None:
		output_video_1.write(frame)

	if output_video_2 is None:	
		fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_2 = cv2.VideoWriter("../output/bird_view.avi", fourcc2, 25,(blank_image.shape[1], blank_image.shape[0]), True)
	elif output_video_2 is not None:
		output_video_2.write(blank_image)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break