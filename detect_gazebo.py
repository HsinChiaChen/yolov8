from ultralytics import YOLO
import rospy
import time
import cv2
import argparse
import numpy as np
from roboflow import Roboflow
from pathlib import Path
from utils.general import increment_path
from pathlib import Path
from my_vision import RobotSensor_vision
from cv_bridge import CvBridge
Vision = RobotSensor_vision()
bridge = CvBridge()

if __name__ == '__main__':
    rospy.init_node('get_goal_point')

    rf = Roboflow(api_key="nFcDZtUvWFGlFx1G4YXi")
    project = rf.workspace().project("gazebo_tree")
    model = project.version(1).model
    print(model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    prev_frame_time = 0
    while not rospy.is_shutdown():
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        print("fps:",fps)

        color_img = Vision.get_color_image()
        color_img = bridge.imgmsg_to_cv2(color_img, "passthrough")
        color_frame = color_img

        img = np.asanyarray(color_frame)
        cv2.imshow("color_img", img)
        cv2.waitKey(1)
        # print(model.predict(img).json())

