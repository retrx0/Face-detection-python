import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.video import VideoStream
import time

video_path = None
if video_path is None:
    vs = VideoStream().start()
else:
    vs = cv2.VideoCapture(video_path)

while True:
    vid_frame = vs.read()
    vid_frame = imutils.resize(vid_frame, width=800)
    # convert to gray scale
    vid_frame_gray = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

    cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    cascade_face_alt = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    cascade_face_alt2 = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
    cascade_eyes = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
    cascade_smile = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

    faces_rects = cascade_face.detectMultiScale(vid_frame_gray, scaleFactor=1.2, minNeighbors=5)
    faces_rect_alt = cascade_face_alt.detectMultiScale(vid_frame_gray, scaleFactor=1.2, minNeighbors=5)
    faces_circle_alt2 = cascade_face_alt.detectMultiScale(vid_frame_gray, scaleFactor=1.2, minNeighbors=5)
    eye_circles = cascade_eyes.detectMultiScale(vid_frame_gray, scaleFactor=1.2, minNeighbors=5)
    smile = cascade_smile.detectMultiScale(vid_frame_gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_rects:
        cv2.rectangle(vid_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
    for (x, y, w, h) in faces_rect_alt:
        # cv2.rectangle(vid_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # top left
        cv2.line(vid_frame, (x, y), (x + int(w / 3), y), (0, 255, 0), 2)
        cv2.line(vid_frame, (x, y), (x, y + int(h / 3)), (0, 255, 0), 2)
        # top right
        cv2.line(vid_frame, (x + (w - int(w / 3)), y), (x + w, y), (0, 255, 0), 2)
        cv2.line(vid_frame, (x + w, y), (x + w, y + (int(h / 3))), (0, 255, 0), 2)
        # bottom left
        cv2.line(vid_frame, (x, y + h), (x + int(w / 3), y + h), (0, 255, 0), 2)
        cv2.line(vid_frame, (x, y + h), (x, y + (h - int(h / 3))), (0, 255, 0), 2)
        # bottom right
        cv2.line(vid_frame, (x + (w - int(w / 3)), y + h), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(vid_frame, (x + w, y + h), (x + w, y + (h - int(h / 3))), (0, 255, 0), 2)

    for (x, y, w, h) in faces_circle_alt2:
        cv2.circle(vid_frame, (x + int(w / 2), y + int(h / 2)), 150, (0, 255, 0), 1)
    for (x, y, w, h) in eye_circles:
        cv2.circle(vid_frame, (x + int(w / 2), y + int(h / 2)), 7, (0, 255, 255), 1)

    cv2.imshow("Face Detector", vid_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(cascade, test_image, scaleFactor=1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    # convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 15)

    return image_copy
