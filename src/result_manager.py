#!/usr/bin/env python
from __future__ import print_function
 
import roslib
import sys
import rospy
import cv2
import os
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped,Point
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd 
import numpy as np
import os
import time
import yaml
import json
import tf
import math
import rospkg
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')


#=================state define==================
st_init =0
st_1 = 1
st_2 = 2
st_3 = 3
st_4 = 4
st_5 = 5
st_6 = 6
st_7 = 7
st_8 = 8

target_object = {}
#===============================================



class node_publisher:
    def __init__(self):
        self.image_pub = rospy.Publisher("/dope/webcam_rgb_raw",Image,queue_size=10)
       

        self.bridge = CvBridge()
  
    def load_image_file(self,filename):
        image_file = cv2.imread(filename)
        return image_file

    def publish_image(self,filename):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.load_image_file(filename), "bgr8"))

class Multi_subscriber:
    def __init__(self,_model,_type):
        self.model = _model
        self.feedback = _type()
        self.flag = False
        self.result_pose_pub = rospy.Publisher("/answer/"+str(self.model),PoseStamped,queue_size=10)

    def sub_cp(self,data):
        self.feedback = data
        self.flag = True

    def get_data(self):
        if self.flag == True:
            # print("return data")
            return self.feedback
        else:
            # print("return None")
            return None

    def shut_flag(self):
        self.feedback=None
        self.flag = False

class result_manager:
    def __init__(self,_params=None,_test_floder=None,_object_number = None, _output_file_name = None):
        self.target={}
        self.sub_cb={}
        self.sub={}
        self.params = _params
        self.test_floder = _test_floder
        self.object_number = str(_object_number)
        
        self.output_file_name = _output_file_name

        self.view_folder = []
        model_path = str()
        model_folder = []
        self.testing_path = str()
        self.filename_list = []
        self.save_path =str()

        self.model_list = self.params['weights']
        self.image_pub = node_publisher()

        self.accuracy = {}
        self.init_sub()
        self.state = st_init
        self.tStart = 0
        self.tEnd = 0
        self.image_index= 0
        self.loca_error = 0
        self.view_num = 0
        self.total_average = pd.DataFrame()


        
    #======================initial subscriber======================
    def init_sub(self):
        for model in self.model_list:
            self.create_sub('/{}/pose_{}'.format(self.params['topic_publishing'], model),model)
            self.accuracy[model] = {"image":[],
                                    "location_error":[],
                                    "roll_error":[],
                                    "pitch_error":[],
                                    "yaw_error":[] } 
        
    def create_sub(self,topic,model):
        self.sub_cb[model]=Multi_subscriber(model,PoseStamped)
        self.sub[model] = \
            rospy.Subscriber(
                topic, 
                PoseStamped, 
                self.sub_cb[model].sub_cp
            )
    #======================initial subscriber======================

    def process(self):
        if self.state == st_init:
            self.view_folder = sorted(os.listdir(self.test_floder))
            model_path = self.test_floder + '/' + self.view_folder[self.view_num]
            model_folder = sorted(os.listdir(model_path))
            self.testing_path = model_path + '/' +model_folder[int(self.object_number)-1]
            print(self.testing_path)
            json_list = [filename for filename in os.listdir(self.testing_path) if filename.endswith('.json')]
            self.filename_list = sorted([os.path.splitext(filename)[0] for filename in json_list])

            self.save_path = str(g_path2package) + "/accuracy/" + str(self.output_file_name) + str(self.view_folder[self.view_num])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.state=st_1
        #--------------------------------------------------------
        elif self.state == st_1:
            image_path = self.testing_path + '/' + self.filename_list[self.image_index]+".png"
            if self.filename_list[self.image_index] == '_camera_settings':
                self.state = st_4
                return
            self.image_pub.publish_image(image_path)
            self.tStart = time.time()#timer start
            self.state = st_2
        #--------------------------------------------------------
        elif self.state == st_2:
            self.tEnd = time.time() #during time
            if self.tEnd- self.tStart < 0:
                return
            self.target = self.get_target()
            if self.target is None:
                self.tEnd = time.time() #during time
                if self.tEnd- self.tStart < 0 :
                    return
            self.state = st_3
        #--------------------------------------------------------
        elif self.state == st_3:
            json_path=self.testing_path + '/' + self.filename_list[self.image_index]+".json"
            json_data=self.loadjson(json_path,self.object_number)
            if self.target is not None:
                for model in self.model_list:
                    if self.target.has_key(str(model)+'_location'):
                        error = self.location_match(self.target[str(model)+"_location"],
                                                          json_data["translations"][0])
                        # print("local_error",error)
                        euler_error = self.quaternion_match(self.target[str(model)+"_euler"],
                                                            json_data['euler_pose'])
                        # print("euler_roll  ",self.target[str(model)+"_euler"][0],json_data['euler_pose'][0])
                        # print("euler_pitch ",self.target[str(model)+"_euler"][1],json_data['euler_pose'][1])
                        # print("euler_yaw   ",self.target[str(model)+"_euler"][2],json_data['euler_pose'][2])
                        self.accuracy[model]["image"].append(self.filename_list[self.image_index])
                        self.accuracy[model]["location_error"].append(error)
                        self.accuracy[model]["roll_error"].append(euler_error[0])
                        self.accuracy[model]["pitch_error"].append(euler_error[1])
                        self.accuracy[model]["yaw_error"].append(euler_error[2])
                        
                        ans = PoseStamped()
                        ans.header.frame_id = '/dope'
                        ans.pose.position.x = json_data["translations"][0][0]
                        ans.pose.position.y = json_data["translations"][0][1]
                        ans.pose.position.z = json_data["translations"][0][2]
                        
                        ans.pose.orientation.x = json_data["quaternion_pose"][0]
                        ans.pose.orientation.y = json_data["quaternion_pose"][1]
                        ans.pose.orientation.z = json_data["quaternion_pose"][2]
                        ans.pose.orientation.w = json_data["quaternion_pose"][3]
                        self.sub_cb[model].result_pose_pub.publish(ans)
            self.image_index=self.image_index +1
            self.state = st_1
        #--------------------------------------------------------
        elif self.state == st_4:
            data_df = pd.DataFrame()
            for model in self.model_list:
                data_df = pd.DataFrame(self.accuracy[model])
                self.total_average[model] = data_df.mean()
                print(data_df.shape[0],len(self.filename_list))
                print(float(data_df.shape[0]/len(self.filename_list)))
                self.total_average.loc["deteced rate"] = data_df.shape[0]/(len(self.filename_list)-2)
                print( self.total_average )
                data_df.loc['avg'] = data_df.mean() 
                data_df.to_csv(str(self.save_path) + '/' + str(model) + ".csv")
                print(str(self.save_path) + '/' + str(model) + ".csv")
            self.state = st_5
            
        #--------------------------------------------------------
        elif self.state == st_5:
            self.total_average.to_csv(str(self.save_path) + "/total_average_"+str(self.object_number) +  "_nd_stage2.csv")
            print(str(self.save_path) + "/total_average_model_" +str(self.object_number) + "_nd_stage2.csv")
            
            self.image_index = 0
            self.view_num = self.view_num +1
            if self.view_num == 1:
                self.state = st_6
                return    

            self.state = st_init
        #--------------------------------------------------------
        #--------------------------------------------------------
        elif self.state == st_6:
            self.state = st_7
            return
        #--------------------------------------------------------
        elif self.state == st_7:
            return
        return
    #============================Tools============================
    def get_target(self):
        target={}
        for model in self.model_list:
            if self.sub_cb[model].get_data() is not None:
                target[str(model)+"_name"] = model
                target[str(model)+"_location"] = self.sub_cb[model].get_data().pose.position
                target[str(model)+"_pose"] = self.sub_cb[model].get_data().pose.orientation
                q = [np.float64(target[str(model)+"_pose"].x),
                     np.float64(target[str(model)+"_pose"].y),
                     np.float64(target[str(model)+"_pose"].z),
                     np.float64(target[str(model)+"_pose"].w)]

                euler_pose = tf.transformations.euler_from_quaternion(q ,"rxyz") 
                target[str(model)+"_euler"] =  np.multiply(euler_pose,(180/math.pi))
            
            self.sub_cb[model].shut_flag()
        if target == {}:
            return None
        return target

    def location_match(self,pd_point,ans_point):
        # print(str(pd_point)+"\n",ans_point)
        return math.sqrt(  (pow(abs(pd_point.x-ans_point[0]),2)) + (pow(abs(pd_point.y-ans_point[1]),2)) + (pow(abs(pd_point.z-ans_point[2]),2))  )
    
    def quaternion_match(self,pd_euler,ans_euler):
        # print(pd_euler)
        # print(ans_euler)
        return abs(pd_euler[0] - ans_euler[0]), abs(pd_euler[1] - ans_euler[1]), abs(pd_euler[2] - ans_euler[2])

    def loadjson(self,path, objectsofinterest):
        """
        Loads the data from a json file. 
        If there are no objects of interest, then load all the objects. 
        """
        with open(path) as data_file:    
            data = json.load(data_file)
        # print (path)
        pointsBelief = []
        boxes = []
        points_keypoints_3d = []
        points_keypoints_2d = []
        pointsBoxes = []
        poses = []
        centroids = []
        
        translations = []
        rotations = []
        points = []

        pose_transfer = []
        cuboid_centroid = []
        euler_pose = []
        for i_line in range(len(data['objects'])):
            info = data['objects'][i_line]
            if not objectsofinterest is None and \
               not objectsofinterest in info['class'].lower():
                continue 

            box = info['bounding_box']
            boxToAdd = []

            boxToAdd.append(float(box['top_left'][0]))
            boxToAdd.append(float(box['top_left'][1]))
            boxToAdd.append(float(box["bottom_right"][0]))
            boxToAdd.append(float(box['bottom_right'][1]))
            boxes.append(boxToAdd)

            boxpoint = [(boxToAdd[0],boxToAdd[1]),(boxToAdd[0],boxToAdd[3]),
                        (boxToAdd[2],boxToAdd[1]),(boxToAdd[2],boxToAdd[3])]

            pointsBoxes.append(boxpoint)

            # 3dbbox with belief maps
            points3d = []

            pointdata = info['projected_cuboid']
            for p in pointdata:
                points3d.append((p[0],p[1]))

            cuboid_centroid = info['cuboid_centroid']
            # Get the centroids
            pcenter = info['projected_cuboid_centroid']

            points3d.append ((pcenter[0],pcenter[1]))
            pointsBelief.append(points3d)
            points.append (points3d + [(pcenter[0],pcenter[1])])
            centroids.append((pcenter[0],pcenter[1]))

            # load translations
            location = info['location']
            translations.append([location[0],location[1],location[2]])
            
            pose_transfer = tf.transformations.quaternion_from_matrix(info['pose_transform'])
            # pose_transfer = tf.transformations.quaternion_inverse(pose_transfer)
            
            # quaternion
            rot = info["quaternion_xyzw"]
            rotations.append(rot)
        
        Rx = tf.transformations.rotation_matrix( 0*math.pi/180, (1, 0, 0))
        Ry = tf.transformations.rotation_matrix(-90*math.pi/180, (0, 1, 0))
        Rz = tf.transformations.rotation_matrix( 90*math.pi/180, (0, 0, 1))
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        
        rot = np.dot(rot,Ry)
        rot = np.dot(rot,Rz)

        # offset_R_q = tf.transformations.quaternion_from_matrix(R)
        # rot = tf.transformations.quaternion_multiply(rot,offset_R_q)
        
        # quaternion_pose = data['camera_data']['quaternion_xyzw_worldframe']
        invers_data_camera_q = tf.transformations.quaternion_inverse(data['camera_data']['quaternion_xyzw_worldframe'])
        quaternion_pose = tf.transformations.quaternion_multiply(rot,invers_data_camera_q)
        # quaternion_pose = tf.transformations.quaternion_multiply(rot,pose_transfer)
            
        
        euler_pose = tf.transformations.euler_from_quaternion(quaternion_pose)
        euler_pose = np.multiply(euler_pose,(180/math.pi))
        
        # euler_pose = tf.transformations.euler_from_quaternion(quaternion_pose ,"rxyz") 
        # euler_pose = np.multiply(euler_pose,(-180/math.pi))
        return {
            "pointsBelief":pointsBelief, 
            "rotations":rotations,
            "translations":translations,
            "centroids":centroids,
            "points":points,
            "keypoints_2d":points_keypoints_2d,
            "keypoints_3d":points_keypoints_3d,
            "cuboid_centroid":cuboid_centroid,
            "euler_pose":euler_pose,
            "quaternion_pose":quaternion_pose,
            }
    #============================Tools============================
def main(args):
    if(len(args)!=4):
        print("-----------------------------------------------------------------------\n" +
              "Wrong args input. The args must be 2 !\n" +  
              "example:\n" +
              "rosrun dope load_image_topic.py testDATA/view1/model_1 1 train_1_v3_vgg-_stage3_60 0 409\n" +
              "arg[1] for #test_folder\n" +
              "arg[2] for #object_number\n" +
              "arg[3] for #output_file_name\n" +
              "arg[4] for #None\n" + 
              "------------------------------------------------------------------------")
        exit(1)
    # __,test_folder,output_file_name,start_filenum,end_filenum = args
    __,test_folder,object_number,output_file_name = args
    test_floder = g_path2package + '/dataset/{}'.format(args[1])
    
    #load yaml file for read
    config_name = "config_result.yaml"
    params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)
    
     #ros part
    rospy.init_node('Result_manager', anonymous=True)
    
    result  = result_manager(params,test_floder,object_number,output_file_name)
    rospy.sleep(.3)

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        try:
            result.process()
        except rospy.ROSInterruptException:
            print('error')
            pass
        rate.sleep()



if __name__ == '__main__':
    main(sys.argv)
    
