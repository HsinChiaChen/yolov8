from ultralytics import YOLO
import rospy
import time
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

from cv_bridge import CvBridge
Vision = RobotSensor_vision()
bridge = CvBridge()

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

    prev_frame_time = 0
    while not rospy.is_shutdown():
        if(cv2.waitKey(1) == ord('q')):
            break
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        print("fps:",fps)

        color_img = Vision.get_color_image()
        color_img = bridge.imgmsg_to_cv2(color_img, "passthrough")
        color_frame = color_img

        img = np.asanyarray(color_frame)
        im0 = img.copy()
        width, height = im0.shape[1], im0.shape[0]
        # cv2.imshow("color_img", img)

        results = model.track(img, persist=True)[0]
        frame_ = results.plot()

        color_edge = Detect_edge(im0)
        img_mask = np.zeros((height, width), dtype=im0.dtype)

        right_line = []
        left_line = []
        for i in range(5):
            right_line.append((width, height))
            left_line.append((0, height))

        print("--------------- New ---------------")
        for box in results.boxes:
            if len(box):
                cls = box.cls[0].item()
                class_id = results.names[cls]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")
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
        color_edge = cv2.add(color_edge, np.zeros(np.shape(im0), dtype=np.uint8), mask=img_mask)

        cv2.imshow('frame', frame_)
        cv2.imshow("color_edge result", color_edge)


