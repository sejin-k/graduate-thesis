# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# 모델 load
# estimator = load_pretrain_model('VGG_origin')
estimator = load_pretrain_model('mobilenet_thin')                                   # openpose model
action_classifier = load_action_premodel('Action/training/fall_detection_v3_input_dim28.h5')
# action_classifier = load_action_premodel('Action/framewise_recognition.h5')     # detect motion

# parm 초기화
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

# 비디오 읽기
cap = choose_run_mode(args)
video_writer = set_video_writer(cap, write_fps=int(7.0))


# # 관절 데이터를 저장하는 txt 파일, training에 사용
# f = open('origin_data.txt', 'a+')
seq_data = []
input_seq_data = []

while cv.waitKey(1) < 0:
    has_frame, show = cap.read()
    if has_frame:
        fps_count += 1
        frame_count += 1

        # pose estimation
        humans = estimator.inference(show)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter.

        if(pose[-1] != []):
            seq_data.append(pose)
            if len(seq_data) == 10:
               input_seq_data = seq_data
        
        #     print("################################ Make input data ################################")
        # # print("################################ check ################################ ")

        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier,input_seq_data)
        if len(seq_data) == 10:
            input_seq_data, seq_data = [], []

        height, width = show.shape[:2]
        # 실시간 FPS 값 표시
        if (time.time() - start_time) > fps_interval:
            # 이 interval 프로세스에서 프레임 수를 계산하고 interval이 1초이면 FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 프레임 수가 매우 적다.
            start_time = time.time()
        # fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        # cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 사람의 수를 표시
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 현재 런타임 및 총 프레임 수 표시
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('Action Recognition based on OpenPose', show)
        video_writer.write(show)

        # # 데이터를 채취하여 훈련 과정에 사용(for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')

video_writer.release()
cap.release()
# f.close()
