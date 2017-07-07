#!/usr/bin/env python

"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from os import listdir
from os.path import isfile, join
from sets import Set


import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import string
import cv2

def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()


def chunks (l, n):
    for i in range(0, len(l), n):
	yield l[i:i + n]

def main(args):
    images, image_file_list = load_data(args.image_files[0], args.image_size, args.margin, args.deblurr,args.gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            name_list = Set()
            for img in image_file_list:
                strs = string.split(img, '/');
                imgName = strs[-1]
                gtName = imgName[:-8]
                name_list.add(gtName)
            
            names = dict()
            count = 1
            for name in name_list:
                names[name] = count
                count = count + 1

            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

	    bLargeList = False

	    emb_list = []
            # Run forward pass to calculate embeddings
	    if (len(images) > 1000):
		bLargeList = True
		images_batch = chunks(images, 1000)
		files_batch  = chunks(image_file_list, 1000)
		for img_list in images_batch:
			for img in img_list:
                		feed_dict = { images_placeholder: [img], phase_train_placeholder:False }
                		temp_emb_list = sess.run(embeddings, feed_dict=feed_dict)
				emb_list.append(temp_emb_list)
		
	    else:
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb_list = sess.run(embeddings, feed_dict=feed_dict)

	    labels_file = args.output_dir[0] + "/" + "labels.csv"
	    reps_file = args.output_dir[0] + "/" + "reps.csv"
	    ptr_l = open(labels_file, 'w')
	    ptr_r = open (reps_file, 'w')

	    count = 0
	    oldName = None
	    for img, emb in zip(image_file_list, emb_list):
        	strs = string.split(img, '/');
	        imgName = strs[-1]
       		gtName = imgName[:-13]
		if bLargeList:
			emb = emb[0]
		myList = ','.join(map(str, emb.tolist()))
	   	ptr_r.write("%s\n" % myList)
		if (gtName == oldName):
			ptr_l.write("%s,%s\n" % (str(count), img))
		else:
			count += 1
			ptr_l.write("%s,%s\n" % (str(count), img))
			oldName = gtName


	    ptr_l.close()
	    ptr_r.close()

            
def load_data(image_paths, image_size, margin, deblurrness, gpu_memory_fraction):
    tempPath = [image_paths + "/" + f for f in listdir(image_paths)]
    img_list = []
    img_file_list = []
    for p in tempPath:
	if (isfile(p)):
		continue
	fileList = [p + "/" + f for f in listdir(p) if isfile(join(p, f))]
	for image_file in fileList:
        	img = misc.imread(image_file, mode='RGB')
		img = misc.imresize(img, (image_size, image_size), interp='bilinear')
        	if (deblurrness > 0):
        	    blurrness = variance_of_laplacian(img)
		    if blurrness < 50:
            		gaussian = cv2.GaussianBlur(img, (19, 19), 20.0)
            		img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
		img = facenet.prewhiten(img)
        	img_list.append(img)
		img_file_list.append(image_file)
    return img_list, img_file_list

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to be embedded')
    parser.add_argument('output_dir', type=str, nargs='+', help='Directory to save CSV file for embedding vector')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--deblurr', type=int, help='DeBlurring face', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
