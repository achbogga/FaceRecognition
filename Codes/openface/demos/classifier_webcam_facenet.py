#!/usr/bin/env python
#
# Example to run classifier on webcam stream.
# Brandon Amos & Vijayenthiran
# 2016/06/21
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Contrib: Vijayenthiran
# This example file shows to run a classifier on webcam stream. You need to
# run the classifier.py to generate classifier with your own dataset.
# To run this file from the openface home dir:
# ./demo/classifier_webcam.py <path-to-your-classifier>


import time

start = time.time()

import argparse
import cv2
import os
import pickle
import tensorflow as tf
import align.detect_face
import random
import dlib

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import facenet

from time import sleep
from scipy import misc

#from gui.webcam import draw_text
from gui import SampleApp


#minsize = 20 # minimum size of face
minsize = 50 # minimum size of face
mtcnn_threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709 # scale factor
mtcnn_rectangle_size_threshold = 100

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def draw_text(frame, text='Please press s to sign up!', x=5, y=50, color=(255,10,10), thickness=2, size=1):
    if x is not None and y is not None:
	cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def InitializeMTCNN ():
    sleep(random.random())
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    return pnet, rnet, onet

def getRep_facenet(bgrImg, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, mtcnn_rectangle_size_threshol, multiple=False):
    start = time.time()
    reps = []
    if multiple:
        bbs = align_dlib.getAllFaceBoundingBoxes(rgbImg)
    else:
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = align.detect_face.detect_face(rgbImg, minsize, pnet, rnet, onet, mtcnn_threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces == 0:
            return []

        for i in range(nrof_faces):
	    det = bounding_boxes[i, 0:4]
            img_size = np.asarray(rgbImg.shape)[0:2]
            margin = 32
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            bb2 = dlib.rectangle(left=int(bb[0]), top=int(bb[1]), right=int(bb[2]), bottom=int(bb[3]))
	    if (bb2.right() - bb2.left() > mtcnn_rectangle_size_threshold):
            	cropped = rgbImg[bb[1]:bb[3],bb[0]:bb[2],:]
            	alignedFace = misc.imresize(cropped, (args.imgDim, args.imgDim), interp='bilinear')
	    	#if (i == 0):
	    	#	cv2.imshow("Face", alignedFace)
	    	#	cv2.waitKey(2)
            	alignedFace = facenet.prewhiten(alignedFace)

            	# Run forward pass to calculate embeddings
            	feed_dict = {images_placeholder: [alignedFace], phase_train_placeholder:False }
            	emb_list = sess.run(embeddings, feed_dict=feed_dict)
            	rep = emb_list[0]
	    else:
		rep = None
            #reps.append((bb2.center().x, rep))
	    reps.append([rep, bb2])

    return reps

def infer(img, args, sess, embeddings, image_placeholder, phase_train_placeholder, pnet, rnet, onet, le, clf, mtcnn_rectangle_size_threshold):

    reps = getRep_facenet(img, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, mtcnn_rectangle_size_threshold)
    #reps = getRep_mtcnn_align(img, sess, embeddings, images_placeholder, phase_train_placeholder, detector)
    persons = []
    confidences = []
    boxes = []
    for rep in reps:
	box = rep[1]
	rep = rep[0]
	if (rep is None):
        	persons.append(None)
        	confidences.append(0.0)
		boxes.append(box)
	else:
        	rep = rep.reshape(1, -1)
        	start = time.time()
        	predictions = clf.predict_proba(rep).ravel()
        	# print predictions
        	maxI = np.argmax(predictions)
        	persons.append(le.inverse_transform(maxI))
        	confidences.append(predictions[maxI])
		boxes.append(box)
    return (persons, confidences, boxes)

def fr_demo(args, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, le, clf):
           
	   # Capture device. Usually 0 will be webcam and 1 will be usb cam.
           video_capture = cv2.VideoCapture(args.captureDevice)
           video_capture.set(3, args.width)
           video_capture.set(4, args.height)
	   cv2.namedWindow('Panel Face Recognition', cv2.WINDOW_NORMAL)
           cv2.setWindowProperty('Panel Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

           repsList = []
           labsList = []
           repsZero = np.zeros([128])
    	   confidenceList = []
	   exit_flag = 1
    	   while True:
        	ret, frame = video_capture.read()
		#print(ret, frame.shape)
        	persons, confidences, boxes = infer(frame, args, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, le, clf, mtcnn_rectangle_size_threshold)
        	print "P: " + str(persons) + " C: " + str(confidences)
        	try:
            		# append with two floating point precision
            		confidenceList.append('%.2f' % confidences[0])
        	except:
            		# If there is no face detected, confidences matrix will be empty.
            		# We can simply ignore it.
            		pass

        	for i, c in enumerate(confidences):
            		if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                		persons[i] = "_unknown"

		width = frame.shape[1]
		frame = cv2.flip(frame, 1)

		for person, confidence, box in zip(persons, confidences, boxes):
			bl = (width - box.right(), box.bottom())
       			tr = (width - box.left(), box.top())

			if (box.right() - box.left() <= mtcnn_rectangle_size_threshold):
	        		cv2.rectangle(frame, bl, tr, color=(0, 255, 0), thickness=1, lineType=100)
				continue

		    	if confidence <= args.threshold:
       		        	name = "Stranger"
       		    	else:
       		        	name = person

			if (box.right() - box.left() > mtcnn_rectangle_size_threshold):
				if (name == "Stranger"):
	        			cv2.rectangle(frame, bl, tr, color=(0, 255, 255), thickness=1, lineType=100)
    					cv2.putText(frame, name, (width - box.right() - 5, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 255), thickness=2, lineType=100)
				else:
	        			cv2.rectangle(frame, bl, tr, color=(255, 0, 255), thickness=1, lineType=100)
    					cv2.putText(frame, name, (width - box.right() - 5, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 0, 255), thickness=2, lineType=100)

        	draw_text(frame)
		cv2.imshow('Panel Face Recognition', frame)
        	# quit the fr_demo on the press of key 's' and begin the demo gui
        	if cv2.waitKey(1) & 0xFF == ord('s'):
            		break
		# quit the program on the press of key 'q'
        	if cv2.waitKey(1) & 0xFF == ord('q'):
		        exit_flag = 0
            		break
	   # When everything is done, release the capture
	   video_capture.release()
	   cv2.destroyAllWindows()
	   return exit_flag

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.", default=os.path.join( dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=160)
    parser.add_argument('--captureDevice', type=int, default=0, help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model_dir', type=str, nargs='+', help="Facenet network model.")
    parser.add_argument('--deblurr', type=int, help='DeBlurring face', default=0)
    parser.add_argument('classifierModel', type=str, help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')


    args = parser.parse_args()

    model_dir = args.model_dir[0]


    align_dlib = openface.AlignDlib(args.dlibFacePredictor)

    #detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

    

    pnet, rnet, onet = InitializeMTCNN()

    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    with tf.Graph().as_default():
        with tf.Session() as sess:

           #print('Model directory: %s' % model_dir)
           #meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
           #print('Metagraph file: %s' % meta_file)
           #print('Checkpoint file: %s' % ckpt_file)
           facenet.load_model(model_dir)

           threshold = args.threshold

           images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
           phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
           embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
	   exit_flag = 1
	   while(exit_flag):
	       exit_flag = fr_demo(args, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, le, clf)
	       if (not exit_flag):
		   break
	       ui = SampleApp('/home/ovuser/FaceRecognition/Codes/openface/demos/db_users.pkl')
	       ui.mainloop()


