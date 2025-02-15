#!/usr/bin/env python
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
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

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import string
import csv
import tensorflow as tf
import align.detect_face
import random


from time import sleep

minsize = 20 # minimum size of face
mtcnn_threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

from sets import Set

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from skimage import transform as transform
from sklearn.calibration import CalibratedClassifierCV
from scipy import misc
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier

from os import listdir

import facenet


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')

classifierAccuracy = 1.0

from itertools import count, takewhile

def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

def frange(start, stop, step):
        return takewhile(lambda x: x< stop, count(start, step))

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


def generateAugmentedImageData (img):
        expImageList = []
        rows, cols, channels = img.shape
        for dy in xrange(-50, 50, 21):
                M = np.float32([[1, 0, 0], [0, 1, dy]])
                dst = cv2.warpAffine(img, M, (cols, rows))
                for angle in xrange(-20, 20, 7):
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        nImg = cv2.warpAffine(dst, M, (cols, rows))
                        expImageList.append(nImg)
        # Create Afine transform
        for s in frange(-0.3, 0.3, 0.1):
                afine_tf = transform.AffineTransform(shear=s)
                # Apply transform to image data
                modified = transform.warp(img, afine_tf)
                expImageList.append(modified)

        return expImageList


def getRep_facenet(imgPath, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, multiple=False):
    start = time.time()
    rgbImg = misc.imread(imgPath, mode='RGB')
    if rgbImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    reps = []
    if multiple:
        bbs, blm_points = align.detect_face.bulk_detect_face(rgbImg, minsize, pnet, rnet, onet, mtcnn_threshold, factor)
    else:
        bounding_boxes, landmark_points = align.detect_face.detect_face(rgbImg, minsize, pnet, rnet, onet, mtcnn_threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces == 0:
                print("Expanded Face Search ...")
                expImageList = generateAugmentedImageData(rgbImg)
                bFound = False
                for expImage in expImageList:
                        bounding_boxes, _ = align.detect_face.detect_face(expImage, minsize, pnet, rnet, onet, mtcnn_threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                                print("Expanded Search Successful !!!")
                                bFound = True
                                rgbImg = expImage
                                break

                if not bFound:
                        print("Expanded Search Fail !!!")
                        return []


        if nrof_faces > 0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(rgbImg.shape)[0:2]
            if nrof_faces>1:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det = det[index,:]
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-args.margin/2, 0)
                            bb[1] = np.maximum(det[1]-args.margin/2, 0)
                            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            #Actual alignment with rotation using landmark points is performed with with this change. -Achyut
                            #bb2 = dlib.rectangle(left=int(bb[0]), top=int(bb[1]), right=int(bb[2]), bottom=int(bb[3]))
                            temp_x = (int(bb[0])+int(bb[2]+1)
                            if (temp_x<0):
                                temp_x-=1
                            temp_y = (int(bb[1])+int(bb[3]+1)
                            if (temp_y<0):
                                temp_y-=1 
                            alignedFace = (extract_image_chips(img,np.transpose(points), args.image_size, 0.37))[0]
            if (args.deblurr > 0):
                blurrness = variance_of_laplacian(alignedFace)
                if (blurrness < 50):
                        print("Sharpening Image...")
                        gaussian = cv2.GaussianBlur(alignedFace, (19, 19), 20.0)
                        alignedFace = cv2.addWeighted(alignedFace, 1.5, gaussian, -0.5, 0)

            alignedFace = facenet.prewhiten(alignedFace)

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: [alignedFace], phase_train_placeholder:False }
            emb_list = sess.run(embeddings, feed_dict=feed_dict)
            rep = emb_list[0]
            reps.append((temp_x/2, rep))

    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

from sklearn.neural_network import MLPClassifier

def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1), map(os.path.split, map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # example-classification-plot-classifier-comparison-py
    # ref: https://jessesw.com/Deep-Learning/
    # i/p nodes, hidden nodes, o/p nodes Smaller steps mean a possibly more accurate result, but the training will take longer
    # a factor the initial learning rate will be multiplied by after each iteration of the training dropouts = 0.25, 
    # Express the percentage of nodes that will be randomly dropped as a decimal.
    # no of iternation 

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, labelsNum)
    accuracy = clf.score(embeddings, labelsNum)
    print("Classifier Accuracy: {}".format(accuracy))
    classifierAccuracy = accuracy 

    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)
    ptr = open(args.acc_output[0], 'w')
    ptr.write("%s\n" % (accuracy))
    ptr.close()
    
def infer(args, multiple=False):
    f = open(args.classifierModel, 'r')
    (le, clf) = pickle.load(f)

    model_dir = args.model_dir[0]

    pnet, rnet, onet = InitializeMTCNN()

    with tf.Graph().as_default():
        with tf.Session() as sess:

    	   print('Model directory: %s' % model_dir)
    	   meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
    	   print('Metagraph file: %s' % meta_file)
    	   print('Checkpoint file: %s' % ckpt_file)
       	   facenet.load_model(model_dir, meta_file, ckpt_file)

           threshold = args.threshold

           pos_count = 0
           neg_count = 0
           sum_count = 0
           not_count = 0
           known_people = Set()

           images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
           phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
           embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

           repsList = []
           labsList = []
           repsZero = np.zeros([128])
           ptr = open(args.output[0], 'w')
           for imgDir in args.imgs:
             imgList = [imgDir + "/" + f for f in listdir(imgDir)]	
	     for img in imgList:
	    	strs = string.split(img, '/');
	        gtName = strs[-2]
		known_people.add(gtName)
	        imgName = strs[-1]
                start = time.time()

       	        #reps = getRep_mtcnn(img, sess, embeddings, images_placeholder, phase_train_placeholder, multiple)
                reps = getRep_facenet(img, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, multiple)
	        if reps == None or reps == []:
		    person = "Undetected"
		    confidence = 0.0
	            print("{:150s}\t{}\t{}\t{}\t{}".format(img, gtName, person, confidence, 0.0)) 
		    ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
		    neg_count += 1
		    not_count += 1
		    repsList.append(repsZero)
		    labsList.append(imgName)
	            continue
                for r in reps:
                    rep = r[1].reshape(1, -1)
                    bbx = r[0]
		    repsList.append(r[1])
		    labsList.append(imgName)
                    predictions = clf.predict_proba(rep).ravel()
                    maxI = np.argmax(predictions)
                    person = le.inverse_transform(maxI)
                    confidence = predictions[maxI]
                    if confidence > threshold:
    			print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, person, confidence, (time.time() - start) * 1000))
    			ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
			if gtName == person:
				pos_count += 1
			else:
				neg_count += 1
		    else:
			print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, "Unknown", confidence, (time.time() - start) * 1000))
			ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, "Unknown", confidence, (time.time() - start) * 1000))
			if gtName == "Unknown":
				pos_count += 1
			else:
				neg_count += 1

           print(known_people)
			
           unknown_people = Set()
           for imgDir in args.unimgs:
            imgList = [imgDir + "/" + f for f in listdir(imgDir)]	
	    for img in imgList:
                strs = string.split(img, '/');
	        gtName = strs[-2]
                imgName = strs[-1]
		unknown_people.add(gtName)
	        gtName = "Unknown"
                start = time.time()
                #reps = getRep_mtcnn(img, sess, embeddings, images_placeholder, phase_train_placeholder, multiple)
                reps = getRep_facenet(img, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, multiple)

                if reps == None or reps == []:
                    person = "Undetected"
                    confidence = 0.0
                    print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, person, confidence, 0.0))
                    ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
		    neg_count += 1
		    not_count += 1
		    repsList.append(repsZero)
		    labsList.append(imgName)
                    continue
                for r in reps:
                    rep = r[1].reshape(1, -1)
                    bbx = r[0]
		    repsList.append(r[1])
		    labsList.append(imgName)
                    predictions = clf.predict_proba(rep).ravel()
                    maxI = np.argmax(predictions)
                    person = le.inverse_transform(maxI)
                    confidence = predictions[maxI]
	            if confidence > threshold:
	    	        print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, person, confidence, (time.time() - start) * 1000))
	        	ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
			if gtName == person:
				pos_count += 1
			else:
				neg_count += 1
	            else:
			print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, "Unknown", confidence, (time.time() - start) * 1000))
			ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, "Unknown", confidence, (time.time() - start) * 1000))
			if gtName == "Unknown":
				pos_count += 1
			else:
				neg_count += 1
    
           ptr.close()

           wild_pos_count = 0
           wild_neg_count = 0
           wild_not_count = 0
           ptr = open(args.wild_output[0], 'w')
           for img in args.wdimgs:
            strs = string.split(img, '/');
            imgName = strs[-1]
	    gtName = imgName[:-8]
	    if gtName not in known_people:
                gtName = "Unknown"
            start = time.time()
            #reps = getRep_mtcnn(img, sess, embeddings, images_placeholder, phase_train_placeholder, multiple)
            reps = getRep_facenet(img, sess, embeddings, images_placeholder, phase_train_placeholder, pnet, rnet, onet, multiple)

            if reps == None or reps == []:
                person = "Undetected"
                confidence = 0.0
                print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, person, confidence, 0.0))
                ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
	        wild_not_count += 1
	        wild_neg_count += 1
	        repsList.append(repsZero)
	        labsList.append(imgName)
                continue
            for r in reps:
                rep = r[1].reshape(1, -1)
                bbx = r[0]
	        repsList.append(r[1])
	        labsList.append(imgName)
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if confidence > threshold:
                    print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, person, confidence, (time.time() - start) * 1000))
                    ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, person, confidence, (time.time() - start) * 1000))
		    if gtName == person:
			wild_pos_count += 1
		    else:
			wild_neg_count += 1
                else:
                    print("{:150s}\t{:10s}\t{:10s}\t{}\t{}".format(img, gtName, "Unknown", confidence, (time.time() - start) * 1000))
                    ptr.write("%s\t%s\t%s\t%f\t%f\n" % (imgName, gtName, "Unknown", confidence, (time.time() - start) * 1000))
		    if gtName == "Unknown":
			wild_pos_count += 1
		    else:
			wild_neg_count += 1

           ptr.close()

           fName = args.output[0] + ".reps.csv"
           ptr = open(fName, 'w')
           for rep in repsList:
	    rep_str = ' '.join(str(e) for e in rep)
	    ptr.write("%s\n" % rep_str)
           ptr.close()

           lName = args.output[0] + ".labels.csv"
           ptr = open(lName, 'w')
           for labs in labsList:
	    ptr.write("%s\n" % labs)
           ptr.close()

    print("Detection: {}".format(pos_count))
    print("NotDetect: {} (NotDetected: {})".format(neg_count, not_count))
    print("Total    : {}".format(pos_count + neg_count))
    print("Accuracy: {}".format(float(pos_count) / float((pos_count + neg_count))))

    print("Wild Images")
    print("Detection: {}".format(wild_pos_count))
    print("NotDetect: {} (NotDetected: {})".format(wild_neg_count, wild_not_count))
    print("Total    : {}".format(wild_pos_count + wild_neg_count))
    print("Accuracy: {}".format(float(wild_pos_count) / float((wild_pos_count + wild_neg_count))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=160)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    trainParser = subparsers.add_parser('train', help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument('--classifier', type=str, choices=[ 'LinearSvm', 'GridSearchSvm', 'GMM', 'RadialSvm', 'DecisionTree', 'GaussianNB', 'DBN'], help='', default='LinearSvm')
    #The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'
    trainParser.add_argument('workDir', type=str, help="")
    trainParser.add_argument('--acc_output', type=str, nargs='+', help="Accuracy Output File.") 
    #trainParser.add_argument('--average_face', type=str, help="Average Face Image.") 

    inferParser = subparsers.add_parser( 'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str, help='Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('--imgs', type=str, nargs='+', help="Input image.") 
    inferParser.add_argument('--unimgs', type=str, nargs='+', help="Unknown image.") 
    inferParser.add_argument('--wdimgs', type=str, nargs='+', help="Wild image.") 
    inferParser.add_argument('--output', type=str, nargs='+', help="Output File.") 
    inferParser.add_argument('--wild_output', type=str, nargs='+', help="Output File for Wild Images.") 
    inferParser.add_argument('--multi', help="Infer multiple faces in image", action="store_true")
    inferParser.add_argument('--threshold', type=float, help="Threshold to classify unknown", default=0.6)
    #inferParser.add_argument('--average_face', type=str, help="Average Face Image.") 
    inferParser.add_argument('--model_dir', type=str, nargs='+', help="Facenet network model.") 
    inferParser.add_argument('--deblurr', type=int, help='DeBlurring face', default=0)


    args = parser.parse_args()

    #print args
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception(""" Torch network model passed as the classification model, which should be a Python pickle (.pkl)""")

    start = time.time()

    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args, args.multi)
