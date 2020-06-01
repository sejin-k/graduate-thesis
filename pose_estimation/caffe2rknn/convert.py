from rknn.api import RKNN

# create RKNN object
rknn = RKNN(verbose=True)

# Load caffe model
print('--> Loading model')
ret = rknn.load_caffe(model='./model/new_model/openpose.prototxt',
					  proto='caffe',
					  blobs='./model/new_model/openpose.caffemodel')
# print error message
if ret !=0:
	print('Load failed!')
	exit(ret)
print('done')


# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)
# print error message
if ret !=0:
	print('Build failed!')
	exit(ret)
print('done')


# Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn('./openpose_caffe.rknn')
# print error message
if ret != 0:
	print('Export failed!')
	exit(ret)
print('done')