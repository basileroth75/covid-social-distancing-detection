import cv2
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("../video/View_001") if isfile(join("../video/View_001", f))]
 
filename_array = []
img_array = []

number_files_start = input("Prompt the starting frame number : ")
number_files_end = input("Prompt the ending frame number : ")


for i in range(int(number_files_start),int(number_files_end)):
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