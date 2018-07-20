
import os, os.path
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import glob
import argparse


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util

parser = argparse.ArgumentParser(description='Get correct file directories')
parser.add_argument('--input_images_path', default = 'input')
parser.add_argument('--output_images_path', default = 'output/')
args = vars(parser.parse_args())

image_list = []
TEST_IMAGE_DIRECTORY = args["input_images_path"]
OUTPUT_DIRECTORY = args["output_images_path"]
path = os.path.join(os.getcwd(),TEST_IMAGE_DIRECTORY)
from os import listdir
from os.path import isfile, join
image_list = [f for f in listdir(path) if isfile(join(path, f))]

MODEL_NAME = 'face_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 1


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


num_detections = detection_graph.get_tensor_by_name('num_detections:0')


for image_file in image_list:
    image = cv2.imread(os.path.join(TEST_IMAGE_DIRECTORY,image_file))
    image_expanded = np.expand_dims(image, axis=0)


    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})


    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)


    cv2.imwrite(OUTPUT_DIRECTORY + image_file,image)
