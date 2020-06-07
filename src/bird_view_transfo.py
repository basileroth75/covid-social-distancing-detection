import numpy as np
import cv2


def compute_perspective_transform(corner_points,width,height):
	""" Compute the transformation matrix
    @ p1,p2,p3,p4 : 4 corner points 
    @ height, width : needed parameter to determinen the matrix
    """
	# Create an array out of theses points
	corner_points_array = np.float32(corner_points)
	# Create an array with the 
	img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
	# Compute the transformation matrix
	return cv2.getPerspectiveTransform(corner_points_array,img_params) 


def compute_point_perspective_transformation(matrix,list_downoids):
	""" Apply the perspective transformation to every poins which have been detected on the main frame.
    @ matrix : the 3x3 matrix 
    @ list_downoids : list that contains the points to transform
    """
	# Compute the new coordinates of our points
	list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
	transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
	# Loop over the points and add them to the list that will be returned
	transformed_points_list = list()
	for i in range(0,transformed_points.shape[0]):
		transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
	return transformed_points_list