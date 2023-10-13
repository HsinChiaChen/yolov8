from ultralytics import YOLO
import rospy
import time
import math
import cv2
import argparse
import numpy as np
from numpy import random
from roboflow import Roboflow
from pathlib import Path
from utils.general import increment_path, scale_coords
from pathlib import Path
from my_vision import RobotSensor_vision
from utils.my_plots import plot_one_box, mask, point_store, draw_line, Detect_edge
from geometry_msgs.msg import Point
import pyrealsense2 as rs


from cv_bridge import CvBridge
Vision = RobotSensor_vision()
bridge = CvBridge()
    
def Dis_goal(depth_frame,width, height, goal_coordinates_array):
    # Get global max depth value
    max_depth = np.amax(depth_frame)
    # print("Max value: {}".format(max_depth))

    # Get global min depth value
    min_depth = np.amin(depth_frame)
    # print("Min value: {}".format(min_depth))

    # Get depth value at a point
    (x_goal, y_goal) = (int(goal_coordinates_array[0][0]), int(goal_coordinates_array[0][1]))
    if depth_frame is not None:
        pixel_distance = depth_frame[int(y_goal*1.5), int(x_goal*2)]*0.001
        # pixel_distance = depth_frame[int(720/2), int(1280/2)]
        # dis = math.sqrt(pixel_distance*pixel_distance - (0.5*0.5))
        print("Distance value: {}m".format(pixel_distance))
        # print((width, height))
        return pixel_distance

    else:
        return None
    
def measure_distance(color_image, depth_image, goal_img_array):
    width, height = color_image.shape[1], color_image.shape[0]
    goal_coordinates_array = []
    
    if depth_frame is not None:
        goal_img_array_len = len(goal_img_array)
        (x_goal, y_goal) = (goal_img_array[0], goal_img_array[1])
        offset_y = x_goal - width / 2
        offset_z = y_goal - height / 2
        # print((x_goal, y_goal))
        # print('offset_y = ', offset_y)
        # print('offset_z = ', offset_z)
        # cv2.circle(color_image, (int(width / 2), int(height / 2)), 10, (0,0,255), 4)
        # print(offset_y*2 + depth_image.shape[1] /2)
        # print(offset_z *1.5 + depth_image.shape[0]/ 2)
        dist =  depth_frame[int(offset_z *1.5 + depth_image.shape[0]/ 2), int(offset_y*2 + depth_image.shape[1] /2)]
        offset_x = dist * 0.001
        # goal_coordinates_array.append([offset_y, offset_z, offset_x])
        print("Distance value: {}m".format(offset_x))
        # goal_coordinates_array.append([offset_x, offset_y, offset_z])
        goal_coordinates_array = (offset_x, offset_y, offset_z)
        return offset_x

    else:
        return None


if __name__ == '__main__':
    rospy.init_node('get_goal_point')

    model = YOLO('best_gazebo.pt')
    model.to('cuda')
    print(model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names  # names {0: 'tree'}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]     # colors [[13, 233, 178]]

    pub = rospy.Publisher("goal_point", Point, queue_size=10)
    rate = rospy.Rate(100) # 10hz

    prev_frame_time = 0
    while not rospy.is_shutdown():
        if(cv2.waitKey(1) == ord('q')):
            break
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        print("fps:",fps)

        depth_img = Vision.get_depth_image()
        depth_img = bridge.imgmsg_to_cv2(depth_img, "passthrough")
        depth_frame = depth_img
        # depth_frame = np.asanyarray(depth_frame)
        # print(depth_frame.shape)
        # cv2.imshow('depth_frame', depth_frame)

        color_img = Vision.get_color_image()
        color_img = bridge.imgmsg_to_cv2(color_img, "passthrough")
        color_frame = color_img
        color_frame = np.asanyarray(color_frame)
        # print(color_frame.shape)

        im0 = color_frame.copy()
        width, height = im0.shape[1], im0.shape[0]
        # cv2.imshow("color_img", img)

        

        results = model.track(color_frame, persist=True)[0]
        frame_ = results.plot()
        goal_img_array = []

        color_edge = Detect_edge(im0)
        img_mask = np.zeros((height, width), dtype=im0.dtype)

        right_line = []
        left_line = []
        for i in range(5):
            right_line.append((width, height))
            left_line.append((0, height))

        # print("--------------- New ---------------")
        for box in results.boxes:
            if len(box):
                cls = box.cls[0].item()
                class_id = results.names[cls]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                # print("Object type:", class_id)
                # print("Coordinates:", cords)
                # print("Probability:", conf)
                # print("---")
                # c = int(class_id)  # integer class
                label = f'{class_id} {conf:.2f}'
                mask(cords,img_mask, color_edge, label=label, color=colors[int(cls)], line_thickness=3)

                [right_line, left_line] = point_store(cords, frame_, names[int(cls)], conf, right_line, left_line)

            else:
                point_size = 10
                point_color_r = (0, 0, 255) # BGR
                thickness = 4 # 可以为 0 、4、8
                y_far = height / 2
                # initial
                y_mid = (height - y_far)*1/3 + y_far
                x_mid = width/2
                (x_goal, y_goal) = (int(x_mid), int(y_mid))
                cv2.circle(frame_, (int(x_mid), int(y_mid)), point_size, point_color_r, thickness)
        
        (x_goal, y_goal) = draw_line(frame_, right_line, left_line)
        # goal_img_array.append([x_goal, y_goal])

        # dist = Dis_goal(depth_frame,width, height, goal_coordinates_array)
        goal_dist = measure_distance(color_frame, depth_frame, (x_goal, y_goal))
        # dis_goal = depth_frame[x_goal, y_goal]
        # print(goal_coordinates_array)

        color_edge = cv2.add(color_edge, np.zeros(np.shape(im0), dtype=np.uint8), mask=img_mask)

        cv2.imshow('frame', frame_)
        # cv2.imshow("color_edge result", color_edge)

        goal_point = Point(x_goal, y_goal, goal_dist)
        # # rospy.loginfo(goal_point)
        # # 將 hello_str 的內容印到螢幕並寫入 ROS log 裡
        pub.publish(goal_point)
        # # 將 hello_str 這個 Message 發佈至 Topic 上
        rate.sleep()

