from rknn.api import RKNN
import cv2 as cv
import numpy as np

img_path = '../image_samples/apink1.jpg'

def convert2inputform(img_path):
	img = cv.imread(img_path, cv.IMREAD_COLOR)
	resized_img = cv.resize(img, dsize=(432, 368), interpolation=cv.INTER_AREA)
	image = cv.dnn.blobFromImage(resized_img, 1.0 / 255, (432, 368), (0, 0, 0), swapRB=False, crop=False)

	return image


def load_rknnfile():
	rknn = RKNN(verbose=True)

	# Direct Load RKNN Model
	# Load rknn model
	rknn.load_rknn('./openpose_caffe.rknn')
	print('--> load success')

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	# print error message
	if ret != 0:
		print('Init runtime environment failed')
	
	return rknn
	

def run_rknn_model(rknn):
	# Inference
	print('--> Running model')
	image = convert2inputform(img_path)
	# input img, runmodel, get output
	outputs = rknn.inference(inputs=[image])
	print('done')
	print(outputs)	# check output

if __name__ == "__main__":
	rknn = load_rknnfile()
	run_rknn_model(rknn)
