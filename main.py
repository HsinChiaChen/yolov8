#################################################

#########    使用yolov8n.pt追蹤farm1.mp4

#################################################
# from ultralytics import YOLO
# import cv2
# # load yolov8 model
# model = YOLO('yolov8n.pt')
# # load video
# video_path = './inference/farm_videos/farm1.mp4'
# cap = cv2.VideoCapture(video_path)
# ret = True
# # read frames
# while ret:
#     ret, frame = cap.read()
#     if ret:
#         # detect objects
#         # track objects
#         results = model.track(frame, persist=True)
#         # plot results
#         # cv2.rectangle
#         # cv2.putText
#         frame_ = results[0].plot()

#         # visualize
#         cv2.imshow('frame', frame_)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
############################################################################################################
#################################################

#########    使用yolov8n.pt追蹤 Webcam (realsense)

#################################################
# from ultralytics import YOLO
# import cv2
# import torch
# import pyrealsense2 as rs
# import numpy as np

# torch.cuda.set_device(0) # Set to your desired GPU number
# # load yolov8 model
# model = YOLO('yolov8n.pt')
# model.to('cuda')
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# pipeline = rs.pipeline()
# profile = pipeline.start(config)
# align_to = rs.stream.color
# align = rs.align(align_to)
# # read frames
# while(True):
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)
#     color_frame = aligned_frames.get_color_frame()
#     depth_frame = aligned_frames.get_depth_frame()
#     if not depth_frame or not color_frame:
#     	continue
	
#     img = np.asanyarray(color_frame.get_data())

#     results = model.track(img, persist=True)

#     frame_ = results[0].plot()
#     # visualize
#     cv2.imshow('frame', frame_)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#     	break


############################################################################################################
#################################################

#########    使用 self model 追蹤 images

#################################################
# from roboflow import Roboflow
# rf = Roboflow(api_key="nFcDZtUvWFGlFx1G4YXi")
# project = rf.workspace().project("gazebo_tree")
# model = project.version(1).model


# # from roboflow import Roboflow
# # rf = Roboflow(api_key="RtIrGWN0ff4BtnhBijdr")
# # project = rf.workspace("cheng-kung-uni").project("tree_gazebo")
# # dataset = project.version(1).download("yolov8")
# # model = project.version(1).model

# # infer on a local image
# # print(model.predict("inference/gazebo_images/gazebo_001.jpg").json())

# response = model.predict("inference/gazebo_images/gazebo_100.jpg", confidence=70).json()

# for pred in response['predictions']:
#     print(pred['class'])

# # infer on an image hosted elsewhere
# # print(model.predict("inference/gazebo_images/").json())

# # save an image annotated with your predictions
# model.predict("inference/gazebo_images/gazebo_051.jpg").save("prediction.jpg")

############################################################################################################
#################################################

#########    建立儲存資料夾

#################################################
# import argparse
# from pathlib import Path
# from utils.general import increment_path
# parser = argparse.ArgumentParser()
# parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# parser.add_argument('--project', default='runs/detect', help='save results to project/name')
# parser.add_argument('--name', default='exp', help='save results to project/name')
# parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# opt = parser.parse_args()


# save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
# (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

# print(save_dir)

############################################################################################################
#################################################

#########    使用yolov8n.pt追蹤farm1.mp4 false

#################################################
# load config
# import json
# with open('roboflow_config.json') as f:
#     config = json.load(f)

#     ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
#     ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
#     ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

#     FRAMERATE = config["FRAMERATE"]
#     BUFFER = config["BUFFER"]

# import cv2
# import base64
# import numpy as np
# import requests
# import time

# # Construct the Roboflow Infer URL
# # (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
# upload_url = "".join([
#     "https://detect.roboflow.com/",
#     ROBOFLOW_MODEL,
#     "?api_key=",
#     ROBOFLOW_API_KEY,
#     "&format=image",
#     "&stroke=5"
# ])

# # Get webcam interface via opencv-python
# video = cv2.VideoCapture(0)

# # Infer via the Roboflow Infer API and return the result
# def infer():
#     # Get the current image from the webcam
#     ret, img = video.read()

#     # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
#     height, width, channels = img.shape
#     scale = ROBOFLOW_SIZE / max(height, width)
#     img = cv2.resize(img, (round(scale * width), round(scale * height)))

#     # Encode image to base64 string
#     retval, buffer = cv2.imencode('.jpg', img)
#     img_str = base64.b64encode(buffer)

#     # Get prediction from Roboflow Infer API
#     resp = requests.post(upload_url, data=img_str, headers={
#         "Content-Type": "application/x-www-form-urlencoded"
#     }, stream=True).raw

#     # Parse result image
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     return image

# # Main loop; infers sequentially until you press "q"
# while 1:
#     # On "q" keypress, exit
#     if(cv2.waitKey(1) == ord('q')):
#         break
    
#     ret, frame = video.read()
#     cv2.imshow('yolov8', frame)

#     # # Capture start time to calculate fps
#     # start = time.time()

#     # # Synchronously get a prediction from the Roboflow Infer API
#     # image = infer()
#     # # And display the inference results
#     # cv2.imshow('image', image)

#     # # Print frames per second
#     # print((1/(time.time()-start)), " fps")

# # Release resources when finished
# video.release()
# cv2.destroyAllWindows()


############################################################################################################
#################################################

#########    listener

#################################################
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point

def callback(data):
    x_goal = data.x
    y_goal = data.y
    z_goal = data.z
    goal_point = (x_goal, y_goal, z_goal)
    print(goal_point)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener')

    rospy.Subscriber("goal_point", Point, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
    

###########################################################################################################
################################################

########    

################################################
# """

# Move to specified pose

# Author: Daniel Ingram (daniel-s-ingram)
#         Atsushi Sakai (@Atsushi_twi)
#         Seied Muhammad Yazdian (@Muhammad-Yazdian)

# P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

# """

# import matplotlib.pyplot as plt
# import numpy as np
# from random import random
# import math


# class PathFinderController:
#     """
#     Constructs an instantiate of the PathFinderController for navigating a
#     3-DOF wheeled robot on a 2D plane

#     Parameters
#     ----------
#     Kp_rho : The linear velocity gain to translate the robot along a line
#              towards the goal
#     Kp_alpha : The angular velocity gain to rotate the robot towards the goal
#     Kp_beta : The offset angular velocity gain accounting for smooth merging to
#               the goal angle (i.e., it helps the robot heading to be parallel
#               to the target angle.)
#     """

#     def __init__(self, Kp_rho, Kp_alpha, Kp_beta):
#         self.Kp_rho = Kp_rho
#         self.Kp_alpha = Kp_alpha
#         self.Kp_beta = Kp_beta

#     def calc_control_command(self, x_diff, y_diff, theta, theta_goal):
#         """
#         Returns the control command for the linear and angular velocities as
#         well as the distance to goal

#         Parameters
#         ----------
#         x_diff : The position of target with respect to current robot position
#                  in x direction
#         y_diff : The position of target with respect to current robot position
#                  in y direction
#         theta : The current heading angle of robot with respect to x axis
#         theta_goal: The target angle of robot with respect to x axis

#         Returns
#         -------
#         rho : The distance between the robot and the goal position
#         v : Command linear velocity
#         w : Command angular velocity
#         """

#         # Description of local variables:
#         # - alpha is the angle to the goal relative to the heading of the robot
#         # - beta is the angle between the robot's position and the goal
#         #   position plus the goal angle
#         # - Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards
#         #   the goal
#         # - Kp_beta*beta rotates the line so that it is parallel to the goal
#         #   angle
#         #
#         # Note:
#         # we restrict alpha and beta (angle differences) to the range
#         # [-pi, pi] to prevent unstable behavior e.g. difference going
#         # from 0 rad to 2*pi rad with slight turn

#         rho = np.hypot(x_diff, y_diff)
#         alpha = (np.arctan2(y_diff, x_diff)
#                  - theta + np.pi) % (2 * np.pi) - np.pi
#         beta = (theta_goal - theta - alpha + np.pi) % (2 * np.pi) - np.pi
#         v = self.Kp_rho * rho
#         w = self.Kp_alpha * alpha - controller.Kp_beta * beta

#         if alpha > np.pi / 2 or alpha < -np.pi / 2:
#             v = -v

#         return rho, v, w


# # simulation parameters
# controller = PathFinderController(9, 15, 3)
# dt = 0.01

# # Robot specifications
# MAX_LINEAR_SPEED = 0.3
# MAX_ANGULAR_SPEED = 0.3

# show_animation = True


# def move_to_pose(x_start, y_start, theta_start, x_goal, y_goal, theta_goal):
#     x = x_start
#     y = y_start
#     theta = theta_start

#     x_diff = x_goal - x
#     y_diff = y_goal - y

#     x_traj, y_traj = [], []

#     rho = np.hypot(x_diff, y_diff)
#     while rho > 0.01:
#         x_traj.append(x)
#         y_traj.append(y)

#         x_diff = x_goal - x
#         y_diff = y_goal - y

#         rho, v, w = controller.calc_control_command(x_diff, y_diff, theta, theta_goal)

#         if abs(v) > MAX_LINEAR_SPEED:
#             v = np.sign(v) * MAX_LINEAR_SPEED

#         if abs(w) > MAX_ANGULAR_SPEED:
#             w = np.sign(w) * MAX_ANGULAR_SPEED

#         theta = theta + w * dt
#         x = x + v * np.cos(theta) * dt
#         y = y + v * np.sin(theta) * dt

#         print("currently:\tlinear vel %s\t angular vel %s " % (v,w))
#         print("currently:\tx %s\t y %s " % (x,y))
#         if show_animation:  # pragma: no cover
#             plt.cla()
#             plt.arrow(x_start, y_start, np.cos(theta_start),
#                       np.sin(theta_start), color='r', width=0.1)
#             plt.arrow(x_goal, y_goal, np.cos(theta_goal),
#                       np.sin(theta_goal), color='g', width=0.1)
#             plot_vehicle(x, y, theta, x_traj, y_traj)


# def plot_vehicle(x, y, theta, x_traj, y_traj):  # pragma: no cover
#     # Corners of triangular vehicle when pointing to the right (0 radians)
#     p1_i = np.array([0.5, 0, 1]).T
#     p2_i = np.array([-0.5, 0.25, 1]).T
#     p3_i = np.array([-0.5, -0.25, 1]).T

#     T = transformation_matrix(x, y, theta)
#     p1 = np.matmul(T, p1_i)
#     p2 = np.matmul(T, p2_i)
#     p3 = np.matmul(T, p3_i)

#     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
#     plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-')
#     plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-')

#     plt.plot(x_traj, y_traj, 'b--')

#     # for stopping simulation with the esc key.
#     plt.gcf().canvas.mpl_connect(
#         'key_release_event',
#         lambda event: [exit(0) if event.key == 'escape' else None])

#     plt.xlim(0, 5)
#     plt.ylim(0, 5)

#     plt.pause(dt)


# def transformation_matrix(x, y, theta):
#     return np.array([
#         [np.cos(theta), -np.sin(theta), x],
#         [np.sin(theta), np.cos(theta), y],
#         [0, 0, 1]
#     ])


# def main():
#     target_pos = np.array([1.0, 2.0])# 定義目標點
#     robot_pos = np.array([0.0, 0.0])# 定義機器人位置

#     x_start = robot_pos[0]
#     y_start = robot_pos[1]
#     theta_start = math.atan2(y_start, x_start)

#     x_goal = target_pos[0]
#     y_goal = target_pos[1]
#     theta_goal = math.atan2(y_goal, x_goal)


#     print("Initial x: %.2f m\nInitial y: %.2f m\nInitial theta: %.2f rad\n" %
#             (x_start, y_start, theta_start))
#     print("Goal x: %.2f m\nGoal y: %.2f m\nGoal theta: %.2f rad\n" %
#             (x_goal, y_goal, theta_goal))
#     move_to_pose(x_start, y_start, theta_start, x_goal, y_goal, theta_goal)

# if __name__ == '__main__':
#     main()


