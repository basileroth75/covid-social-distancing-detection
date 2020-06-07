import cv2
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("../video/View_001") if isfile(join("../video/View_001", f))]
 
filename_array = []
img_array = []

number_files = input("Prompt the number of frames you want to use for the video : ")

for i in range(0,int(number_files)):
	number = ""
	if i < 10:
		number = "00"+str(i)
	elif i<100:
		number = "0"+str(i)
	else :
		number = str(i)
	filename = "../video/View_001/frame_0"+number+".jpg"
	filename_array.append(filename)
 

for filename in filename_array:
	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width,height)
	img_array.append(img)

fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
output_video_2 = cv2.VideoWriter("../video/pets2009.avi", fourcc2, 25,(width, height), True)

for i in range(len(img_array)):
	output_video_2.write(img_array[i])