#!/usr/bin/env python
from __future__ import print_function
 
import roslib
import sys
import rospy
import cv2
import os
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd 

 
class image_converter:
 
  def __init__(self):
    self.image_pub = rospy.Publisher("/dope/webcam_rgb_raw",Image,queue_size=10)
    self.all_peaks_sub = rospy.Subscriber("/dope/all_peaks_cracker", Float32MultiArray, self.all_peaks_callback)

    self.bridge = CvBridge()
    self.all_peaks = []
  
  def load_image_file(self,filename):
    image_file = cv2.imread(filename)
    return image_file

  def publish_image(self,filename):
    self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.load_image_file(filename), "bgr8"))

  def all_peaks_callback(self,all_peaks_topic):
    self.all_peaks = all_peaks_topic.data
  def get_peaks(self):
    return self.all_peaks
 
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  rate = rospy.Rate(4)

  now_work_path = os.getcwd()
  print(now_work_path)
  # filepath = now_work_path + "/src/dope/dope_objects"
  filepath = now_work_path + "/src/dope/dataset/data_v2/sofa/"
  filenum = 0
  filetype = ".png"

  accuracy ={"filename":[],"peak_0":[],"peak_1":[],"peak_2":[],"peak_3":[],"peak_4":[],"peak_5":[],"peak_6":[],"peak_7":[],"peak_8":[]} 
  
  for i in range(410):
    filename = filepath + '{:06d}'.format(filenum+i) + filetype
    # print(filename)
    # if raw_input("in " + '{:06d}'.format(filenum+i) + ".png Countinus... ")!= 'n':
    if rospy.is_shutdown():
      break
    else:
      ic.publish_image(filename)
      # ic.publish_image(filepath+filetype)
      rate.sleep()
      all_peaks = ic.get_peaks()
      accuracy["filename"].append(filenum+i)
      for j in range(len(all_peaks)):
        accuracy["peak_" + str(j)].append(all_peaks[j])
    cv2.waitKey(0)
  print(len(accuracy["filename"]))
  for i in range(8):
    print(len(accuracy["peak_"+str(i)]))

  data_df = pd.DataFrame(accuracy)
  data_df.loc['avg'] = data_df.mean()
  print("accuracy is :",data_df.loc['avg'])

  data_df.to_csv(str(now_work_path) + "/src/dope/src/accuracy.csv")
  print("save file accuracy.csv in " + str(now_work_path) + "/src/dope/src/")

if __name__ == '__main__':
    main(sys.argv)
