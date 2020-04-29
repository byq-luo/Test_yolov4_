#from keras_yolo4.yolo import Yolo4

import cv2
from yolo import Yolo4
from PIL import Image
import matplotlib.pyplot as plt


model_path = 'yolo4_weight.h5'
anchors_path = 'model_data/yolo4_anchors.txt'
classes_path = 'model_data/coco_classes.txt'

score = 0.5
iou = 0.5
model_image_size = (608, 608)
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

while True:
    img = input("Input image: ")

    if img == "q":
    	break

    else:
	    try:
	        image = Image.open("images_test/test{}.jpg".format(img))
	    except:
	        print("Open Error.!, Try again..")
	        continue

	    else:
	        image = yolo4_model.detect_image(image, model_image_size=model_image_size)
	        plt.imshow(image)
	        plt.show()
	        
    if cv2.waitKey(1) & 0xFF == ord("q"):
    	break

cv2.destroyAllWindows()