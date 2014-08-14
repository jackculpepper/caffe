import numpy as np
import cv2
import os
from math import floor, ceil
import ctypes
import pickle
from skimage import transform

class Detector(object):

  face_cascade = cv2.CascadeClassifier('koestinger_cascade_aflw_lbp.xml')
  #profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

  out_dim = (200, 250)
  crop_height_scale = .80
  crop_width_scale = .40

  def __init__(self, **kwargs):
    pass

  def detect_faces(self, img):
    #img_rr = rotateImage(img, -15)
    #img_lr = rotateImage(img, 15)

    faces = self.face_cascade.detectMultiScale(img, 1.1, 3,
    	cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
    #faces_rr = face_cascade.detectMultiScale(img_rr, 1.1, 3,
    	#cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
    #faces_lr = face_cascade.detectMultiScale(img_lr, 1.1, 3,
    	#cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
    #profile = profile_cascade.detectMultiScale(gray, 1.1, 3,
    	#cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
    #profile_r = profile_cascade.detectMultiScale(gray, 1.1, 3,
    	#cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))

    print "Detected " + str(len(faces)) + " faces"
    #print "Detected " + str(len(faces_rr)) + " faces_rr"
    #print "Detected " + str(len(faces_lr)) + " faces_lr"
    #print "Detected " + str(len(profile)) + " profile"
    # print "Detected " + str(len(profile_r)) + " profile_r"

    return faces

  def crop_faces(self, img, faces):
    img_faces = []

    for (x, y, w, h) in faces:
      y_lower = max(0, y-ceil(h*self.crop_height_scale))
      y_upper = min(img.shape[0], y+h*(1+self.crop_height_scale))
      x_lower = max(0, x-ceil(h*self.crop_width_scale))
      x_upper = min(img.shape[1], x+w*(1+self.crop_width_scale))

      crop = img[y_lower:y_upper, x_lower:x_upper]
      crop = cv2.resize(crop, self.out_dim, interpolation=cv2.INTER_AREA)
      img_faces.append(crop)

    return img_faces

  #def save_faces(img):
	#		#cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)

  def rotateImage(image, angle):
    (_, _, channels) = image.shape

    result = []
    for c in range(channels):
      image_center = tuple(np.array(image[:,:,c].shape)/2)
      rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
      result.append(cv2.warpAffine(image[:,:,c], rot_mat,
    image[:,:,c].shape[::-1], flags=cv2.INTER_LINEAR))

    return np.dstack(result)
