import numpy as np
import cv2

# Load image
image = cv2.imread("frame_from_video.jpg")

width,height,_ = image.shape

p2 = [1879 ,  228 ] # TOP LEFT
p1 = [ 1022 ,  91 ] # BOTTOM LEFT
p4 = [1637 ,  1072] # TOP RIGHT 
p3 = [9 ,  883] # BOTTOM RIGHT

list_points_to_detect = [[1361,438],[994,525]]

"""
p2 = [1637 ,  1072] TOP RIGHT 
p3 = [9 ,  883] BOTTOM RIGHT
p4 = [ 1022 ,  91 ] BOTTOM LEFT
p1 = [1879 ,  228 ] TOP LEFT
"""

'''
cv2.circle(image, p1, 1, (0, 0, 255), -1)
cv2.circle(image, p2, 1, (0, 0, 255), -1)
cv2.circle(image, p3, 1, (0, 0, 255), -1)
cv2.circle(image, p4, 1, (0, 0, 255), -1)

cv2.putText(image,"p1",p1,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA) 
cv2.putText(image,"p2",p2,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA) 
cv2.putText(image,"p3",p3,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA) 
cv2.putText(image,"p4",p4,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA) 
'''


corner_points_array = np.float32([p1,p2,p3,p4])
img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
# Compute the transformation matrix
matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
# show the 4 points on the picture 
for x in range (0,4):
    cv2.circle(image,(corner_points_array[x][0],corner_points_array[x][1]),15,(255,0,0),cv2.FILLED)
for pt in list_points_to_detect:
    cv2.circle(image,(pt[0],pt[1]),8,(0,255,0),cv2.FILLED)


# Compute the new coordinates of our points
list_points_to_detect = np.float32(list_points_to_detect).reshape(-1, 1, 2)
transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
# Compute the warped image with the matrix
imgOutput = cv2.warpPerspective(image,matrix,(width,height))
# loop over the points
for i in range(0,transformed_points.shape[0]):
	x = transformed_points[i][0][0]
	y = transformed_points[i][0][1]
	cv2.circle(imgOutput,(x,y),20,(0,255,0),cv2.FILLED)


# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Output Image ", imgOutput)
cv2.waitKey(0)

