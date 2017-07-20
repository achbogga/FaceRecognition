#!/usr/bin/env python2
import numpy as np
import string
import pickle
import os
import argparse
import shutil
import time

from os import listdir
from os.path import isfile, join
from shutil import copyfile
from subprocess import call

import glob
import cv2

from cv2 import imread


def setupEvaluation (mypath, dumpPath):
	tempPath = mypath + "/" + "OVI-Result"
	if not os.path.exists(tempPath):
		os.makedirs(tempPath)
	trainPath = mypath + "/" + "OVI-Train"
	testPath = mypath + "/" + "OVI-Test"
	facePath = mypath + "/../../OVI-FaceData"

	dumpTrainPath = dumpPath + "/" + "OVI-Train"
	if not os.path.exists(dumpTrainPath):
		os.makedirs(dumpTrainPath)

	fileList = [trainPath + "/" + f for f in listdir(trainPath) if isfile(join(trainPath, f))]
        for fileName in fileList:
                trainName = fileName.split('/')[-1].split('.')[0]

                tmpName = trainPath + "/" + "TrainSet"
                if not os.path.exists(tmpName):
                       os.makedirs(tmpName)
                tmpName = trainPath + "/" + "TrainSet" + "/" + trainName
                if not os.path.exists(tmpName):
                        os.makedirs(tmpName)

                tmpName = dumpTrainPath + "/" + "TrainSet"
                if not os.path.exists(tmpName):
                       os.makedirs(tmpName)
                tmpName = dumpTrainPath + "/" + "TrainSet" + "/" + trainName
                if not os.path.exists(tmpName):
                        os.makedirs(tmpName)

                rawName = dumpTrainPath + "/" + "TrainSet" + "/" + trainName + "/raw"
                if not os.path.exists(rawName):
                        os.makedirs(rawName)

                alignedPath = dumpTrainPath + "/" + "TrainSet" + "/" + trainName + "/Aligned"
	 	if not os.path.exists(alignedPath):
			os.makedirs(alignedPath)

                featurePath = trainPath + "/" + "TrainSet" + "/" + trainName + "/Feature"
		if not os.path.exists(featurePath):
			os.makedirs(featurePath)

                with open(fileName, 'r') as myfile:
                        for line in myfile.readlines():
				strs = string.split(line, '\\');
				dirName = strs[0]
				imgName = strs[1]
				line = string.replace(line, '\\', '/')
				line = string.replace(line, '\r', '\0')
				line = string.replace(line, '\n', '\0')
				imgName = string.replace(imgName, '\r', '')
				imgName = string.replace(imgName, '\n', '')
				dirName = string.replace(dirName, '\r', '')
				dirName = string.replace(dirName, '\n', '')
				scrName = facePath + "/" + dirName + "/" + imgName
                                tmpName = dumpTrainPath + "/" + "TrainSet" + "/" + trainName + "/raw"
				dstName = tmpName + "/" + dirName
				if not os.path.exists(dstName):
					os.makedirs(dstName)
				shutil.copy2(scrName, dstName)


	fileList = [testPath + "/" + f for f in listdir(testPath) if isfile(join(testPath, f))]
        for fileName in fileList:
                with open(fileName, 'r') as myfile:
                        for line in myfile.readlines():
                                strs = string.split(line, '\\');
                                dirName = strs[0]
                                imgName = strs[1]
                                line = string.replace(line, '\\', '/')
                                line = string.replace(line, '\r', '\0')
                                line = string.replace(line, '\n', '\0')
                                imgName = string.replace(imgName, '\r', '')
                                imgName = string.replace(imgName, '\n', '')
                                dirName = string.replace(dirName, '\r', '')
                                dirName = string.replace(dirName, '\n', '')
				scrName = facePath + "/" + dirName + "/" + imgName

				testName = fileName.split('/')[-1].split('.')[0]
                                tmpName = testPath + "/" + "TestSet"
                                if not os.path.exists(tmpName):
                                        os.makedirs(tmpName)
                                tmpName = testPath + "/" + "TestSet" + "/" + testName
                                if not os.path.exists(tmpName):
                                        os.makedirs(tmpName)
                                dstName = testPath + "/" + "TestSet" + "/" + testName + "/" + dirName
                                if not os.path.exists(dstName):
                                        os.makedirs(dstName)
                                shutil.copy2(scrName, dstName)

def runEvaluation (mypath, dumppath, threshold, testOnly):
        trainPath = mypath + "/" + "OVI-Train/TrainSet"
        dumpTrainPath = dumppath + "/" + "OVI-Train/TrainSet"
        testPath  = mypath + "/" + "OVI-Test"
        resultPath  = mypath + "/" + "OVI-Result"
	UnknownPath = testPath + "/TestSet/unknown/*"

        tempPath = [trainPath + "/" + f for f in listdir(trainPath)]
        dumpTempPath = [dumpTrainPath + "/" + f for f in listdir(dumpTrainPath)]

	runFile = "/FR/Codes/openface/demos/classifier_facenet_mtcnn.py"
	wildPath = mypath + "/../../overflow"
	networkPath = "/FR/models/20170512-110547" 

        for p, d in zip(tempPath, dumpTempPath):
		alignedPath = d + "/" + "Aligned"
		featurePath = p + "/" + "Feature"
		rawPath = d + "/" + "raw"
		tmpName = p.split('/')[-1]
		evalPath = testPath + "/TestSet/" + tmpName + "/*"
		outputPath = resultPath + "/" + tmpName + "-openface-thld-" + str(threshold) + ".txt"
		wild_outputPath = resultPath + "/" + tmpName + "-openface-set-aside-thld-" + str(threshold) + ".txt"
		logPath = resultPath + "/" + tmpName + "-openface-thld-" + str(threshold) + ".log"
		accPath = resultPath + "/" + tmpName + "-openface-thld_acc.log"
		#avgFacePath = d + "/" + tmpName + "-average-face.tif"

		start_time = time.time()
		if not testOnly:
			command = "/FR/Codes/facenet/src/align/align_dataset_mtcnn_expand.py " + rawPath + " " + alignedPath + "  --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.4 " 
			print(command)
			os.system(command)

			command = "/FR/Codes/facenet/src/facenet_embedding.py " + networkPath + " " + alignedPath + " " + featurePath + " --image_size 160 --deblurr 0"
			print(command)
			os.system(command)


		command = runFile + " train " + featurePath + " --acc_output " + accPath
		print(command)
		os.system(command)
				
		end_time = time.time()
		ptr = open(logPath, 'w')
		ptr.write("%f" % ((end_time - start_time) * 1000.0))
		ptr.close()

		command = runFile + " infer " +  " --threshold " + str(threshold) + " " + featurePath + "/classifier.pkl " 
		command += "--model_dir " + networkPath
		command += " --imgs " + evalPath + " --unimgs " + UnknownPath + " --output " + outputPath + " --wdimgs " + wildPath + "/*"
		#command += " --wild_output " + wild_outputPath + " --average_face " + avgFacePath
		command += " --wild_output " + wild_outputPath + " --deblurr 0 "
		print(command)
		os.system(command)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('workDir', type=str, help="")
	args = parser.parse_args()
	print(args)

	threshold = 0.0
	dumpPath = args.workDir + "/CV-Temp"
	if os.path.exists(dumpPath):
                shutil.rmtree(dumpPath)
		os.makedirs(dumpPath)
        else:
                os.makedirs(dumpPath)
	mypath = args.workDir + "/CV-Groups"
        tempPath = [mypath + "/" + f for f in listdir(mypath)]
        for p in tempPath:
		if os.path.isdir(p):
			pathName = p.split('/')[-1]
			tPath = dumpPath + "/" + pathName 
			print(tPath)
			if not os.path.exists(tPath):
				os.makedirs(tPath)

			setupEvaluation(p, tPath)
			runEvaluation(p, tPath, threshold, False)

