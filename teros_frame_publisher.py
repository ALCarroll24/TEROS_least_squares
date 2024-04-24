#!/usr/bin/env python3

import rospy
import rospkg
import yaml
import os
import numpy as np
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray
import tf2_ros
from tf.transformations import quaternion_from_euler
from scipy.optimize import least_squares

class RobotDetectionNode:
    def __init__(self):
        rospy.init_node('teros_frame_publisher')

        # Parameters
        self.estimate = rospy.get_param('~estimate', False)
        self.map_is_origin = rospy.get_param('~map_is_origin', False)
        yaml_filename = rospy.get_param('~yaml_file', 'transforms.yaml')
        
        # Find path from this ros package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('yolo_gazebo')
        self.yaml_file = os.path.join(package_path, 'config', yaml_filename)

        # Subscribers
        rospy.Subscriber("/gv01/detections", Detection2DArray, self.gv01_detection_callback)
        rospy.Subscriber("/gv02/detections", Detection2DArray, self.gv02_detection_callback)

        # TF2 broadcaster
        self.br = tf2_ros.TransformBroadcaster()

        # Initialize accumulation variables if needed
        self.accumulated_positions = {'gv01': [], 'gv02': []}
        
        # Save detections from each vehicle as they come in
        self.last_jeep_detection = None
        self.last_wolf_detection = None
        
        # If we are estimating
        if self.estimate:
            # Timer for estimating the transform
            rate = 1 # Hz
            self.estimation_timer = rospy.Timer(rospy.Duration(1/rate), self.estimate_and_save_transform)
            
            # Pair the last detections from each vehicle at a rate
            rate = 10 # Hz
            self.save_detection_timer = rospy.Timer(rospy.Duration(1/rate), self.save_detection_pair)
            
        # Timer for publishing the transforms (just zeros when estimating)
        rate = 1000
        self.publish_transform_timer = rospy.Timer(rospy.Duration(1/rate), self.publish_transform)
            
        # Spin so the node stays alive and manages callbacks
        rospy.spin()

    def gv01_detection_callback(self, msg):
        self.last_jeep_detection = msg
    
    def gv02_detection_callback(self, msg):
        self.last_wolf_detection = msg

    def save_detection_pair(self, _):
        jeep = self.last_jeep_detection
        wolf = self.last_wolf_detection
        
        # Check that we have our first message for both of these
        if jeep is None or wolf is None:
            rospy.loginfo_throttle(0.5, "Waiting for first detection pair")
            return
        
        # Make sure we have a detection in each message
        if len(jeep.detections) == 0 or len(wolf.detections) == 0:
            return
        if len(jeep.detections[0].results) == 0 or len(wolf.detections.results) == 0:
            return
        
        # Pull out the field with the data
        jeep_result = jeep.detections[0].results[0]
        wolf_result = jeep.detections[0].results[0]
        
        jeep_det_pos = jeep_result.pose.pose.position.x, jeep_result.pose.pose.position.y
        wolf_det_pos = wolf_result.pose.pose.position.x, wolf_result.pose.pose.position.y
        
        self.accumulated_positions['gv01'].append(jeep_det_pos)
        self.accumulated_positions['gv02'].append(wolf_det_pos)
        rospy.loginfo("Adding detection pair to accumulated positions")

    def estimate_and_save_transform(self, _):
        # Example of processing; actual computation would depend on your application
        if len(self.accumulated_positions['gv01']) < 10:  # arbitrary number to ensure enough data
            rospy.loginfo("Not enough data to estimate transform")
            return
            
        # Initial guesses for the hidden variables
        initial_guess = self.load_transform # D, theta1, theta2

        # Perform least squares optimization
        result = least_squares(
            self.error_function, initial_guess,
            args=(self.accumulated_positions['gv01'], self.accumulated_positions['gv02'])
        )

        # The optimized hidden variables
        optimized_D, optimized_theta1, optimized_theta2 = result.x
        
        # Save the estimate to yaml file
        with open(self.yaml_file, 'w') as file:
            yaml.dump([optimized_D, optimized_theta1, optimized_theta2], file)
            
    def error_function(self, hidden_variables, detections_jeep, detections_wolf):
        D, theta1, theta2 = hidden_variables
        errors = []
        
        for (x_jeep, y_jeep), (x_wolf, y_wolf) in zip(detections_jeep, detections_wolf):
            
            det_jeep = np.array([x_jeep, y_jeep])
            det_wolf = np.array([x_wolf, y_wolf])
            
            # Move the Wolf detections into Jeep space using hidden variables
            rot_wolf = self.rotate(det_wolf, -theta2)
            translated_wolf = np.array(rot_wolf[0] - D, rot_wolf[1])
            transformed_wolf = self.rotate(translated_wolf, theta1)
            
            # Calculate error in x and y
            error_x = det_jeep[0] - transformed_wolf[0]
            error_y = det_jeep[1] - transformed_wolf[1]
            
            errors.extend([error_x, error_y])
                
    def rotate(self, vec, degrees):
        R = np.array([[np.cos(np.deg2rad(degrees)), -np.sin(np.deg2rad(degrees))],
                        [np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees))]])
        
        return R @ vec
                
    def load_transform(self):
        # Load transforms from a YAML file
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as file:
                return yaml.load(file, Loader=yaml.FullLoader)
    
    def publish_transform(self, _):
        # Get the transform variables
        D, theta1, theta2 = self.load_transform()
        
        # If we weren't able to load to transform
        if D is None or theta1 is None or theta2 is None:
            rospy.log_warn("No transform loaded, unable to publish")
            return
        
        # Calculate quaternians for each rotation
        q1 = quaternion_from_euler(0, 0, np.deg2rad(theta1))
        q2 = quaternion_from_euler(0,0, np.deg2rad(theta2))
        
        # If we are estimating then we want detections to be in base_link frame, so we make the transformation 0
        if self.estimate:
            D = 0
            theta1 = 0
            theta2 = 0
            q1 = np.array([0,0,0,1])
            q2 = np.array([0,0,0,1])
        
        if self.map_is_origin:
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = 'map'
            transform.child_frame_id = 'base_link'
            transform.transform.translation.x = 0
            transform.transform.translation.y = 0
            transform.transform.translation.z = 0
            transform.transform.rotation.x = q1[0]
            transform.transform.rotation.y = q1[1]
            transform.transform.rotation.z = q1[2]
            transform.transform.rotation.w = q1[3]
            self.br.sendTransform(transform)
        else:
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = 'map'
            transform.child_frame_id = 'base_link'
            transform.transform.translation.x = D
            transform.transform.translation.y = 0
            transform.transform.translation.z = 0
            transform.transform.rotation.x = q2[0]
            transform.transform.rotation.y = q2[1]
            transform.transform.rotation.z = q2[2]
            transform.transform.rotation.w = q2[3]
            self.br.sendTransform(transform)
            
        rospy.loginfo_throttle(0.5, "Published transforms")

    def load_transforms(self):
        # Load transforms from a YAML file
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        else:
            return {}

if __name__ == '__main__':
    node = RobotDetectionNode()
