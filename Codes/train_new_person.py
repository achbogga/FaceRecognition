#!/usr/bin/env python
import video_to_faces
import os
import shutil

def train_new_person(user_folder)
	data_dir = '/home/ovuser/FaceRecognition/Data/OVI-CV-03-Facenet'
	temp_dir = '/home/ovuser/FaceRecognition/Data/OVI-CV-03-Facenet/temp'
	align_folder_code = '/home/ovuser/FaceRecognition/Codes/facenet/src/align/align_folder.py'
	embed_code = '/home/ovuser/FaceRecognition/Codes/facenet/src/facebet_embedding.py'
	network_path = '/home/ovuser/FaceRecognition/models/20170512-110547'
	classifier_path = '/home/ovuser/FaceRecognition/Codes/openface/demos/classifier_facenet_mtcnn.py'
	training_log_file = '/home/ovuser/FaceRecognition/Data/OVI-CV-03-Facenet/training.log'
	username = user_folder.replace('/home/ovuser/FaceRecognition/db/', '')
	aligned_folder = os.path.join(data_dir + '/OVI-FaceData-aligned-160', username)
	full_dataset_aligned = '/home/ovuser/FaceRecognition/Data/OVI-CV-03-Facenet/OVI-FaceData-aligned-160'
	
	for video is os.listdir(user_folder):
		if (os.path.exists(temp_dir)):
			shutil.rmtree(temp_dir)
			os.makedirs(temp_dir)
		#convert video to frames
		video_to_faces.video_to_faces(os.path.join(user_folder,video), temp_dir)
	
	#perform alignment
	os.system(align_folder_code+" "+temp_dir+" "+aligned_folder)	
	#calculate embeddings
	os.system(embed_code+" "+network_path+" "+full_dataset_aligned+"--image_size 160 --margin 32 --deblurr 0")
	#train the classifier
	os.system(classifier_path+" train "+data_dir)
	
