# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from pathlib import Path
from Tracking.deep_sort import preprocessing
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.detection import Detection
from Tracking import generate_dets as gdet
from Tracking.deep_sort.tracker import Tracker
from keras.models import load_model
from .action_enum import Actions

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

# 기본 변수 정의
file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
input_seq = 10

# deep_sort 초기화
model_filename = str(file_path/'Tracking/graph_model/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box color(green)
trk_clr = (0, 255, 0)


# class ActionRecognizer(object):
#     @staticmethod
#     def load_action_premodel(model):
#         return load_model(model)
#
#     @staticmethod
#     def framewise_recognize(pose, pretrained_model):
#         frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
#         joints_norm_per_frame = np.array(pose[-1])
#
#         if bboxes:
#             bboxes = np.array(bboxes)
#             features = encoder(frame, bboxes)
#
#             # score to 1.0 here).
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
#
#             # 进行非极大抑制
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]
#
#             # 调用tracker并实时更新
#             tracker.predict()
#             tracker.update(detections)
#
#             # 记录track的结果，包括bounding boxes及其ID
#             trk_result = []
#             for trk in tracker.tracks:
#                 if not trk.is_confirmed() or trk.time_since_update > 1:
#                     continue
#                 bbox = trk.to_tlwh()
#                 trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
#                 # 标注track_ID
#                 trk_id = 'ID-' + str(trk.track_id)
#                 cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)
#
#             for d in trk_result:
#                 xmin = int(d[0])
#                 ymin = int(d[1])
#                 xmax = int(d[2]) + xmin
#                 ymax = int(d[3]) + ymin
#                 # id = int(d[4])
#                 try:
#                     # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
#                     # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
#                     tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
#                     j = np.argmin(tmp)
#                 except:
#                     # 若当前帧无human，默认j=0（无效）
#                     j = 0
#
#                 # 进行动作分类
#                 if joints_norm_per_frame.size > 0:
#                     joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
#                     joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
#                     pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
#                     init_label = Actions(pred).name
#                     # 显示动作类别
#                     cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
#                 # 画track_box
#                 cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
#         return frame

def load_action_premodel(model):
    return load_model(model)


def framewise_recognize(pose, pretrained_model,seq_data=[]):
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    joints_norm_per_frame = np.array(pose[-1])

    input_datas = []
    if len(seq_data) == input_seq:
        for p in seq_data:
            input_datas.append(np.array(p[-1])) 

    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        # 비극대적 억제를 하다
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # 트랙커를 호출하여 실시간으로 업데이트
        tracker.predict()
        tracker.update(detections)

        # 트랙의 결과를 기록하고 bounding boxes와 그 ID를 포함
        trk_result = []
        for trk in tracker.tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh()
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
            # track_ID 표시
            trk_id = 'ID-' + str(trk.track_id)
            cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                # xcenter는 한 이미지에서 모든 휴먼의 1번 관절점(neck)의 x 좌표 값입니다
                # 트랙_박스와 휴먼의 xcenter 사이의 거리 계산을 통해 ID 매칭
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                # 현재 프레임에 휴먼이 없으면 j=0
                j = 0

            
            # 진행 동작 분류
            if joints_norm_per_frame.size > 0:
                joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                init_label = "moving"

                if len(input_datas) == input_seq:
                    rnnInput = []
                    for input_data in input_datas:
                        joints_data = input_data[j*36:(j+1)*36]
                        joints_data = np.array(joints_data).reshape(-1, 36)
                        print(joints_data[:,:28].shape)
                        rnnInput.append(joints_data[:,:28])

                    
                    rnnInput = np.array(rnnInput)
                    rnnInput = rnnInput.reshape(-1,10,28)
                    
                    pred = np.argmax(pretrained_model.predict(rnnInput))
                    init_label = Actions(pred).name

                # 넘어짐 표시
                cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
                
                # 이상 경보(under scene)
                if init_label == 'fall_down':
                    cv.putText(frame, 'fall Detection!!!!', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 255), 4)
            # track_box
            cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)

    return frame


################################################################################################################################
# # -*- coding: UTF-8 -*-
# import numpy as np
# import cv2 as cv
# from pathlib import Path
# from Tracking.deep_sort import preprocessing
# from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
# from Tracking.deep_sort.detection import Detection
# from Tracking import generate_dets as gdet
# from Tracking.deep_sort.tracker import Tracker
# from keras.models import load_model
# from .action_enum import Actions

# # Use Deep-sort(Simple Online and Realtime Tracking)
# # To track multi-person for multi-person actions recognition

# # 기본 변수 정의
# file_path = Path.cwd()
# clip_length = 15
# max_cosine_distance = 0.3
# nn_budget = None
# nms_max_overlap = 1.0

# # deep_sort 초기화
# model_filename = str(file_path/'Tracking/graph_model/mars-small128.pb')
# encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# tracker = Tracker(metric)

# # track_box color(green)
# trk_clr = (0, 255, 0)


# # class ActionRecognizer(object):
# #     @staticmethod
# #     def load_action_premodel(model):
# #         return load_model(model)
# #
# #     @staticmethod
# #     def framewise_recognize(pose, pretrained_model):
# #         frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
# #         joints_norm_per_frame = np.array(pose[-1])
# #
# #         if bboxes:
# #             bboxes = np.array(bboxes)
# #             features = encoder(frame, bboxes)
# #
# #             # score to 1.0 here).
# #             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
# #
# #             # 进行非极大抑制
# #             boxes = np.array([d.tlwh for d in detections])
# #             scores = np.array([d.confidence for d in detections])
# #             indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
# #             detections = [detections[i] for i in indices]
# #
# #             # 调用tracker并实时更新
# #             tracker.predict()
# #             tracker.update(detections)
# #
# #             # 记录track的结果，包括bounding boxes及其ID
# #             trk_result = []
# #             for trk in tracker.tracks:
# #                 if not trk.is_confirmed() or trk.time_since_update > 1:
# #                     continue
# #                 bbox = trk.to_tlwh()
# #                 trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
# #                 # 标注track_ID
# #                 trk_id = 'ID-' + str(trk.track_id)
# #                 cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)
# #
# #             for d in trk_result:
# #                 xmin = int(d[0])
# #                 ymin = int(d[1])
# #                 xmax = int(d[2]) + xmin
# #                 ymax = int(d[3]) + ymin
# #                 # id = int(d[4])
# #                 try:
# #                     # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
# #                     # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
# #                     tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
# #                     j = np.argmin(tmp)
# #                 except:
# #                     # 若当前帧无human，默认j=0（无效）
# #                     j = 0
# #
# #                 # 进行动作分类
# #                 if joints_norm_per_frame.size > 0:
# #                     joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
# #                     joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
# #                     pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
# #                     init_label = Actions(pred).name
# #                     # 显示动作类别
# #                     cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
# #                 # 画track_box
# #                 cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
# #         return frame

# def load_action_premodel(model):
#     return load_model(model)


# def framewise_recognize(pose, pretrained_model):
#     frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
#     joints_norm_per_frame = np.array(pose[-1])

#     if bboxes:
#         bboxes = np.array(bboxes)
#         features = encoder(frame, bboxes)

#         # score to 1.0 here).
#         detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

#         # 비극대적 억제를 하다
#         boxes = np.array([d.tlwh for d in detections])
#         scores = np.array([d.confidence for d in detections])
#         indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
#         detections = [detections[i] for i in indices]

#         # 트랙커를 호출하여 실시간으로 업데이트
#         tracker.predict()
#         tracker.update(detections)

#         # 트랙의 결과를 기록하고 bounding boxes와 그 ID를 포함
#         trk_result = []
#         for trk in tracker.tracks:
#             if not trk.is_confirmed() or trk.time_since_update > 1:
#                 continue
#             bbox = trk.to_tlwh()
#             trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
#             # track_ID 표시
#             trk_id = 'ID-' + str(trk.track_id)
#             cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)

#         for d in trk_result:
#             xmin = int(d[0])
#             ymin = int(d[1])
#             xmax = int(d[2]) + xmin
#             ymax = int(d[3]) + ymin
#             # id = int(d[4])
#             try:
#                 # xcenter는 한 이미지에서 모든 휴먼의 1번 관절점(neck)의 x 좌표 값입니다
#                 # 트랙_박스와 휴먼의 xcenter 사이의 거리 계산을 통해 ID 매칭
#                 tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
#                 j = np.argmin(tmp)
#             except:
#                 # 현재 프레임에 휴먼이 없으면 j=0
#                 j = 0

#             # 진행 동작 분류
#             if joints_norm_per_frame.size > 0:
#                 joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
#                 joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
#                 joints_norm_single_person = joints_norm_single_person[:,:28]
#                 pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
#                 init_label = Actions(pred).name
#                 # 显示动作类别
#                 cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
#                 # 异常预警(under scene)
#                 if init_label == 'fall_down':
#                     cv.putText(frame, 'WARNING: someone is falling down!', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
#                                1.5, (0, 0, 255), 4)
#             # track_box
#             cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
#     return frame

