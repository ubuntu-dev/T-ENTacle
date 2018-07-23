#! /usr/bin/python
# coding: utf-8

# Object Detection Demo
# Welcome to the object detection inference walkthrough!
# This notebook will walk you step by step through the process of using a pre-trained model to
# detect objects in an image. Make sure to follow the [installation instructions]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
# before you start.

# Imports

from flask import Flask, request, Response, jsonify
import json,decimal
import requests
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import os
from io import BytesIO
import urllib
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

app = Flask(__name__)

# Model preparation
# Variables
#
# Using previously exported model 
MODEL_NAME = 'model_24062018'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/app/model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/app', 'table_label_map.pbtxt')

# Only class used - Table
NUM_CLASSES = 1

# ## Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open('/app/detection_model.tar.gz')
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that
# returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Detection
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(0, 7) ]
IMAGE_SIZE = (12, 8)    # Size, in inches, of the output images.


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def inference(image_data):
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Returned output_dict contains float32 numpy arrays
    box_list = output_dict['detection_boxes'].tolist()
    # box = output_dict['detection_boxes'][0]
    box = box_list[0]
    ymin_normal = box[0]
    xmin_normal = box[1]
    ymax_normal = box[2]
    xmax_normal = box[3]

    confidence_list = output_dict['detection_scores'].tolist()
    confidence_score = confidence_list[0]


    ymin_pixels = box[0] * height
    xmin_pixels = box[1] * width
    ymax_pixels = box[2] * height
    xmax_pixels = box[3] * width

    
    detection_list = output_dict['detection_classes'].tolist()
    detection_class = detection_list[0]

    response = {
        'box-pixels': {
            'top-left-y': ymin_pixels,
            'top-left-x': xmin_pixels,
            'bottom-right-y': ymax_pixels,
            'bottom-right-x': xmax_pixels
        },
        'box-normal': {
            'top-left-y': ymin_normal,
            'top-left-x': xmin_normal,
            'bottom-right-y': ymax_normal,
            'bottom-right-x': xmax_normal
        },
        'confidence': confidence_score,
        'class': detection_class
    }
    return response


def get_remote_file(url, timeout=10):
    try:
        result = requests.get(url, stream=True, timeout=timeout)
        if result.status_code == 200:
            return result.headers.get('Content-Type', 'application/octet-stream'), result.raw.data
    except requests.exceptions.Timeout:
        timeout=20
    except requests.exceptions.RequestExceptions as e:
        pass

    return None, None
    


@app.route("/")
def main():
    content = "<html>" \
                "<body>" \
                    "<h1>Welcome to Table Detection Inference</h1>" \
                    "<p>Usage:</p>" \
                    "<p>https://localhost:8764/detect/table/url=1.jpg</p>" \
                "</body>" \
              "</html>"

    return content


# URL
@app.route("/detect/table", methods=["GET", "POST"])
def detect():
    if request.method == 'POST':
        image_data = request.get_data()
    else:
        url = request.args.get("url")
        image_type, image_data = get_remote_file(url)

        if not image_data:
            return Response(status=404)
   
    response = inference(image_data)

    return Response(response=json.dumps(response), status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="9009")
