#!/usr/bin/env python
from __future__ import print_function
 
import roslib
import sys
import rospy
import cv2
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
 
class image_converter:
 
  def __init__(self):
    self.image_pub = rospy.Publisher("/dope/webcam_rgb_raw",Image,queue_size=10)
    self.bridge = CvBridge()
  
  def load_image_file(self,filename):
    image_file = cv2.imread(filename)
    return image_file

  def publish_image(self,filename):
    self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.load_image_file(filename), "bgr8"))
 
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)

  now_work_path = os.getcwd()
  print(now_work_path)
  # filepath = now_work_path + "/src/dope/dope_objects"
  filepath = now_work_path + "/src/dope/dataset/data_v2/sofa/"
  filenum = 100
  filetype = ".png"

  for i in range(500):
    filename = filepath + '{:06d}'.format(filenum+i) + filetype
    # print(filename)
  # while True:
    # filename = filepath + "000347" + filetype
    if raw_input("in " + '{:06d}'.format(filenum+i) + ".png Countinus... ")!= 'n':
      try:
        ic.publish_image(filename)
        # ic.publish_image(filepath+filetype)
      except KeyboardInterrupt:
        print("Shutting down") 
        break
    else:
      break
    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv)
