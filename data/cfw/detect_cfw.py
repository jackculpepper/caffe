"""
detect_cfw.py
Detect, crop and align faces in the CFW dataset
"""
import numpy as np
import cv2
import os
from math import floor, ceil
import ctypes
import pickle
from skimage import transform

from detector import Detector
from aligner import Aligner

SRC_DIR = './cfw_full/'
DST_DIR = './cfw_full_vj_test/'

from IPython.parallel import Client
from IPython import parallel
rc = Client()
dview = rc[:]

def main():
  CELEB_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR))
  dview.push(dict(SRC_DIR=SRC_DIR, DST_DIR=DST_DIR))
  return dview.map_async(align_celeb_images, CELEB_LIST)

def align_celeb_images(celeb):
  import numpy as np
  import cv2
  import os
  from math import floor, ceil
  import ctypes
  import pickle
  from skimage import transform

  from aligner import Aligner
  from detector import Detector

  kwargs = {'stasm_path':'../../../stasm4.1.0/build/libstasm.so',
    'data_path':'../../../stasm4.1.0/data'}
  aligner = Aligner(**kwargs)
  detector = Detector()

  celeb_string = celeb.replace(' ', '_')

  if not os.path.exists(DST_DIR + celeb_string):
    os.makedirs(DST_DIR + celeb_string)

  IMAGE_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR + celeb))

  for image in IMAGE_LIST:
    img = cv2.imread(SRC_DIR + celeb + '/' + image)

    if img == None:
      print "Invalid image. Deleting image"
      os.remove(SRC_DIR + celeb + '/' + image)
      continue

    faces = detector.detect_faces(img)
    img_faces = detector.crop_faces(img, faces)

    for j, img_face in enumerate(img_faces):
      img_face_path = DST_DIR + celeb_string + '/' + os.path.splitext(image)[0] \
        + '_' + str(j) + '.jpg'

      #cv2.imshow(str(j) + 'a', aligner.draw_points_on_face(point_list,
      #    img_face.copy()))

      img_face, end_pose = aligner.align(img_face, img_face_path)
      cv2.imwrite(img_face_path, img_face)

    print SRC_DIR + celeb + '/' + image
