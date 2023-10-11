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

#########    使用yolov8n.pt追蹤farm1.mp4 false

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
    
