
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

BASE_DIR= os.getcwd()
floating_model = False

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(BASE_DIR, 'labelmap.pbtxt')

NUM_CLASSES = 1

print('PATH_TO_LABELS=',PATH_TO_LABELS)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

TF_MODEL=  'output_tflite_graph_helmet.tflite'
interpreter = tf.lite.Interpreter(model_path=TF_MODEL)

interpreter.allocate_tensors()

input_mean=127.5
input_std=127.5

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
if input_details[0]['dtype'] == np.float32:
  floating_model = True

height = input_details[0]['shape'][1]   
width = input_details[0]['shape'][2]

print("height=", height)
print("width=", width)

input_shape = input_details[0]['shape']

video = cv2.VideoCapture(1)
ret = video.set(3,1280)
ret = video.set(4,720)
while(True):
    ret, frame = video.read()
    # Test model on random input data.
    if ret:
       
        image_np_x = cv2.resize(frame, (height,width))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        input_data = np.expand_dims(image_np_x, axis=0)
        if floating_model:
          input_data = (np.float32(input_data) - input_mean) / input_std
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])
        if(num > 0):
            #print('------------')
            #print('num',num)
            #print('classes=',classes[0])
            #print('scores=',scores[0])
            for i in range(num):
              classes[0][i]=classes[0][i] + 1.0
              id = int(classes[0][i])
              sco=scores[0][i]
              name='none'
              if(id in category_index):
                s=category_index[id]
                name=s['name']
              #print(name,'/',sco)
                
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.4,       
                max_boxes_to_draw=num,      
                line_thickness=1)
        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break
# Clean up
video.release()
cv2.destroyAllWindows()
