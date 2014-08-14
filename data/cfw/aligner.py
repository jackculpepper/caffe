import numpy as np
import cv2
import os
from math import floor, ceil
import ctypes
import pickle
from skimage import transform

from detector import Detector

class Aligner(object):
  # target face size to align to
  kOutputFaceSize = 200, 250

  # number of landmarks returned by stasm
  kStasmNLandmarks = 77

  # file containing the default points in default pose
  #kDefaultDataFile = 'default_pose.pickle'

  # similarity threshold for comparing two poses
  kSimilarityThreshold = 1.0

  # indices of the landmark points we want
  kLeftEyeLCornerIndex = 34
  kLeftEyeRCornerIndex = 30
  kRightEyeLCornerIndex = 40
  kRightEyeRCornerIndex = 44
  kNoseLeftIndex = 58
  kNoseTipIndex = 52
  kNoseRightIndex = 54
  kMouthLeftIndex = 59
  kMouthRightIndex = 65
  marker_indices = [kLeftEyeLCornerIndex, kLeftEyeRCornerIndex,
                    kRightEyeLCornerIndex, kRightEyeRCornerIndex,
                    kNoseLeftIndex, kNoseTipIndex, kNoseRightIndex,
                    kMouthLeftIndex, kMouthRightIndex]

  def __init__(self, **kwargs):
    # load stasm dynamically
    self.stasmlib = ctypes.cdll.LoadLibrary(kwargs['stasm_path'])

    # get the face detector data path
    self.data_path = kwargs['data_path']

    # initialize stasm
    self.initialized = self.stasmlib.stasm_init(self.data_path, ctypes.c_int(0))

    # load default pose
    # self.load_default_pose('default_face.pickle')

    if self.initialized:
      self.foundface = ctypes.c_int(0)
      self.landmarks = (ctypes.c_float * 2 * self.kStasmNLandmarks)()
    else:
      print 'STASM initialization failed: data_path = ' + self.data_path

    self.default_pose = self.process_default_pose('./default_face.jpg',
            './default_pose.pickle')

  def get_interest_points(self):

    def center_point(point_list):
      # Averages eye L\R and mouth L\R
      return [avg_tuple(point_list[0], point_list[1]), avg_tuple(
        point_list[2], point_list[3]), avg_tuple(point_list[-2],
        point_list[-1])]

    def avg_tuple(a, b):
      return (ceil((a[0]+b[0])/2.0), ceil((a[1]+b[1])/2.0))

    point_list = []
    for index in self.marker_indices:
        p = (self.landmarks[index][0], self.landmarks[index][1])
        point_list.append(p)

    point_list = center_point(point_list)

    return point_list

  def process_face(self, image_data, image_path):
    assert self.initialized

    # load the src image and convert to grayscale
    # note: opencv always opens a grayscale image into 3 channels by default,
    # we'll need to do the conversion in that case as well
    channels = image_data.shape[2]
    assert channels == 3 or channels == 4
    if channels == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGRA2GRAY)

    success = self.stasmlib.stasm_search_single(
                ctypes.byref(self.foundface),
                ctypes.cast(self.landmarks, ctypes.POINTER(ctypes.c_float)),
                image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
                ctypes.c_int(image_data.shape[1]),
                ctypes.c_int(image_data.shape[0]),
                image_path, self.data_path)

    return self.get_interest_points()

  def process_default_pose(self, image_path, data_file_path):
    point_list = []
    image = cv2.imread(image_path)
    detector = Detector()
    faces = detector.detect_faces(image)
    img_faces = detector.crop_faces(image, faces)
    point_list = self.process_face(img_faces[0], image_path)
    #cv2.imshow('Default pose', self.draw_points_on_face(point_list,
    #    img_faces[0]))

    with open(data_file_path, 'w') as fh:
        pickle.dump(point_list, fh)
    return point_list

  def load_default_pose(self, data_file_path):
      point_list = []
      with open(data_file_path, 'r') as fh:
          point_list = pickle.load(fh)
      return point_list

  def align(self, face_image, image_path):
    start_pose = self.process_face(face_image, image_path)
    t = transform.SimilarityTransform()
    t.estimate(np.array(self.default_pose), np.array(start_pose))
    warped_image = np.array(transform.warp(face_image, t)*255, dtype=np.uint8)
    end_pose = self.process_face(warped_image, image_path)

    return warped_image, end_pose

  def draw_points_on_face(self, point_list, face_image):
    assert self.initialized
    result = face_image
    for p in point_list:
      cv2.circle(result, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    return result

  def final_crop(self, image):
      x, y = self.kOutputFaceSize
      center_x = ceil(image.shape[1]/2.0)
      center_y = ceil(image.shape[0]/2.0)
      return image[center_y-y/2:center_y+y/2, center_x-x/2:center_x+x/2]

