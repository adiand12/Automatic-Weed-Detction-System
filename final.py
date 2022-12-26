from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import matplotlib.image as mpimg
import keras
import tensorflow as tf
import serial
from serial import Serial
import time
arduino = serial.Serial(port='COM6', baudrate=115200, timeout=.1)
i = 1
labelsPath = 'data/names/obj.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = 'data/weights/' + 'crop_weed_detection.weights'
configPath = 'data/cfg/crop_weed.cfg'
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
confi = 0.5
thresh = 0.5
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
model = load_model('Code/models/UNet.h5')
IMG_SIZE = 512
IMG_SIZE = 250


def decode_image_from_raw_bytes(filename):
    # img = cv2.imread(filename)
    img = cv2.cvtColor(filename, cv2.COLOR_BGR2RGB)
    return img


def image_center_crop(img):
    cropped_img = img[int((img.shape[0] - min(img.shape[0], img.shape[1]))/2): int((img.shape[0] + min(img.shape[0], img.shape[1]))/2),
                      int((img.shape[1] - min(img.shape[0], img.shape[1]))/2): int((img.shape[1] + min(img.shape[0], img.shape[1]))/2), :]

    return cropped_img


def prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=True):
    img = image_center_crop(raw_bytes)  # take squared center crop
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize for our model
    if normalize_for_model:
        img = img.astype("float32")  # prepare for normalization
        img = tf.keras.applications.inception_v3.preprocess_input(
            img)  # normalize for model
    return img


cap = cv2.VideoCapture(0)
# focus = 0  # min: 0, max: 255, increment:5
# cap.set(28, focus)
# # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
# # cap.set(3, 1280) # set the Horizontal resolution
# # cap.set(4, 720) # Set the Vertical resolution
# ret, image = cap.read()

# if not ret:
#     print("failed to grab frame")
#     break
# image = cv2.imread('Dataset/weed_4.jpeg')
# cv2.imshow('Capturing Video', image)
while(1):
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
    # cap.set(3, 1280)  # set the Horizontal resolution
    # cap.set(4, 720)  # Set the Vertical resolution
    ret, image = cap.read()
    cv2.imshow('Capturing Video', image)
    # ret, image = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    # image = cv2.imread('Dataset/30238eb8-4e6c-4c3b-81e8-b576d9f4d5b6.JPG')
    # cv2.imshow('Capturing Video', image)
    img = decode_image_from_raw_bytes(image)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confi:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
    if len(idxs) > 0:
        print("yes weed detected!")
        x = '1'
    else:
        print("No")
        x = '0'
    arduino.write(bytes(x, 'utf-8'))
    if len(idxs) > 0:
        IMG_SIZE = 512
        img = prepare_raw_bytes_for_model(img, normalize_for_model=False)
        test = np.reshape(img, (-1, IMG_SIZE, IMG_SIZE, 3))
        predictions = model.predict(test)
        predictions = np.reshape(predictions, (IMG_SIZE, IMG_SIZE))
        predictions = np.round(predictions)
        i = i+1
        mpimg.imsave(f"out_{i}.png", predictions)
        # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(3)
cap.release()
cv2.destroyAllWindows()
