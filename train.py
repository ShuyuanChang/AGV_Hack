#!/usr/bin/python
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

GPIO.setmode(GPIO.BCM)
PWM_PIN1 = 17 
PWM_PIN2 = 18
GPIO.setup(PWM_PIN1,GPIO.OUT)
GPIO.setup(PWM_PIN2,GPIO.OUT)
pwm1 = GPIO.PWM(PWM_PIN1,500)
pwm2 = GPIO.PWM(PWM_PIN2,500)
pwm1.start(0)
pwm2.start(0)

# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

graph_def = tf.GraphDef()
labels = []

# Import the TF graph
with tf.gfile.FastGFile('model.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open('labels.txt', 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (227, 227), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image



duty_s = raw_input("Enter Duty Cycle (0 to 100):")
duty = int(duty_s)
if duty >= 0 and duty <=100 :           
	pwm1.ChangeDutyCycle(duty)
	pwm2.ChangeDutyCycle(duty)

try:
    while True:
        
        image = vs.read()
        image = imutils.resize(image, width=400)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imgfilestr = '/home/pi/logs/SnapshotTest-' + timestr +'.jpg'
        cv2.imwrite(imgfilestr,image)
        time.sleep(0.1)

			
except KeyboardInterrupt:
		pwm1.stop()
		pwm2.stop()
		GPIO.cleanup()    
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()
