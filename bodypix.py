import os
import numpy as np
import tensorflow as tf
import cv2
import math
from PIL import Image,ImageFilter,ImageOps
from utils import load_graph_model, get_input_tensors, get_output_tensors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

OutputStride = 16
modelPath = './bodypix_mobilenet_float_050_model-stride16/model.json'
#modelPath = './bodypix_resnet50_float_model-stride16/model.json'

print("Loading model...", end="")
graph = load_graph_model(modelPath)
print("done.\nLoading sample image...", end="")

#capture = cv2.VideoCapture("out.mp4")
capture = cv2.VideoCapture(0)

input_tensor_names = get_input_tensors(graph)
output_tensor_names = get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

sess = tf.compat.v1.Session(graph=graph)

while(capture.isOpened()):
	ret, image = capture.read()
	InputImageShape = image.shape
	targetWidth = (InputImageShape[1] // OutputStride) * OutputStride + 1
	targetHeight = (InputImageShape[0] // OutputStride) * OutputStride + 1
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image = cv2.resize(image,(targetWidth, targetHeight))
	InputImageShape = image.shape

	widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
	heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1

	if any('resnet_v1' in name for name in output_tensor_names):
		# add imagenet mean - extracted from body-pix source
		m = np.array([-123.15, -115.90, -103.06])
		x = np.add(image, m)
	# For Mobilenet
	elif any('MobilenetV1' in name for name in output_tensor_names):
		x = (image/127.5)-1
	else:
		print('Unknown Model')

	sample_image = x[tf.newaxis, ...]
	results = sess.run(output_tensor_names, feed_dict={input_tensor: sample_image})

	for idx, name in enumerate(output_tensor_names):
		if 'displacement_bwd' in name:
			pass
		elif 'displacement_fwd' in name:
			pass
		elif 'float_heatmaps' in name:
			heatmaps=np.squeeze(results[idx],0)
		elif 'float_long_offsets' in name:
			longoffsets=np.squeeze(results[idx],0)
		elif 'float_short_offsets' in name:
			offsets=np.squeeze(results[idx],0)
		elif 'float_part_heatmaps' in name:
			partHeatmaps=np.squeeze(results[idx],0)
		elif 'float_segments' in name:
			segments=np.squeeze(results[idx],0)
		elif 'float_part_offsets' in name:
			partOffsets=np.squeeze(results[idx],0)
		else:
			print('UnknownOutputTensor',name,idx)
	# Segmentation MASk
	segmentation_threshold = 0.1
	segmentScores = tf.sigmoid(segments)
	mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
	segmentationMask = tf.dtypes.cast(mask, tf.int8)
	segmentationMask = np.reshape(segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))

	# Draw Segmented Output
	mask_img = Image.fromarray(segmentationMask * 255)
#	mask_img = mask_img.resize((targetWidth*2, targetHeight*2), Image.LANCZOS).convert("RGB")
#	mask_img = mask_img.resize((targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
	mask_img = mask_img.resize((targetWidth, targetHeight), Image.BOX).convert("RGB")
	mask_img = tf.keras.preprocessing.image.img_to_array(mask_img, dtype=np.uint8)
#	segmentationMask_inv = np.bitwise_not(mask_img)
	fg = np.bitwise_and(np.array(image), np.array(mask_img))
	cv2.imshow("camera",pil2cv(fg))
#	cv2.imshow("camera",pil2cv(mask_img))
	if cv2.waitKey(10) > 0:
		break
sess.close()
capture.release()
cv2.destroyAllWindows()
