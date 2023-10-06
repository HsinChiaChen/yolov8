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