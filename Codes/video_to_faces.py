#!/usr/bin/env python
import cv2
import numpy as np
from scipy import misc
import os

#working_dir = os.getcwd()
#out_dir = os.path.join(working_dir, 'aligned')

def video_to_faces(video_src, output_dir):
	if (not os.path.exists(output_dir)):
		os.makedirs(output_dir)
	#cap = cv2.VideoCapture(1)
	l = len(os.listdir(output_dir))	
	cap = cv2.VideoCapture(video_src, cv2.CAP_FFMPEG)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if (length>0):
		for i in range(length):
			ret, frame = cap.read()
			if (ret and (frame is not None)):
				rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				rgbImg = misc.imresize(rgbImg, (480,640,3))
				output_file = os.path.join(output_dir, '_'+str(l)+'.png')
				l+=1
				misc.imsave(output_file, rgbImg)
		return 1
	else:
		return 0
	cap.release()
	cv2.destroyAllWindows()
#if __name__ == "__main__":
#   video_to_faces('test.avi', 'test_images')
