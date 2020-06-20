import os
import sys
import natsort
import cv2
import numpy as np
import pandas as pd

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

#################################################### Change parm ####################################################
# openpose Model path
protoFile = "/home/sejin/workspace/graduate-thesis/pose_estimation/caffe2rknn/model/new_model/openpose.prototxt"
weightsFile = "/home/sejin/workspace/graduate-thesis/pose_estimation/caffe2rknn/model/new_model/openpose.caffemodel"

# synch 맞추기 위한 변수
folder_path = "../dataset/frames/chute"
num_of_cam = 8
num_of_folder = 24
save_path = "../dataset/csvfile/"

# Frame information data [start, end, state]
state_frame = np.array([[[874,1011,1],[1012,1079,6],[1080,1108,2],[1109,1285,3]],
               [[308,374,1],[375,399,2],[400,600,3]],
               [[380,590,1],[591,625,2],[626,784,3]],
               [[230,287,1],[288,314,2],[315,380,3],[381,600,6],[601,638,2],[639,780,3]],
               [[288,310,1],[311,336,2],[337,450,3]],
               [[325,582,1],[583,629,2],[630,750,3]],
               [[330,475,1],[476,507,2],[508,680,3]],
               [[144,270,1],[271,298,2],[299,380,3]],
               [[310,472,1],[473,505,5],[506,576,7],[577,627,6],[628,651,2],[652,760,3]],
               [[315,461,1],[462,511,5],[512,530,2],[531,680,3]],
               [[378,463,1],[464,489,2],[490,600,3]],
               [[355,604,1],[605,653,2],[654,770,3]],
               [[301,430,1],[431,476,5],[477,525,7],[526,636,5],[637,717,8],[718,780,6],[781,822,6],[823,863,2],[864,960,3]],
               [[372,555,1],[556,590,5],[591,856,8],[857,934,6],[935,988,6],[989,1023,2],[1024,1115,3]],
               [[363,486,1],[487,530,5],[531,630,7],[631,754,6],[755,787,2],[788,970,3]],
               [[380,455,1],[456,488,5],[489,530,4],[531,568,6],[569,629,5],[630,645,4],[646,670,6],[671,731,5],[732,817,7],[818,890,6],[891,940,2],[941,1000,3]],
               [[251,315,1],[316,340,5],[341,361,4],[362,388,6],[389,410,5],[411,430,4],[431,460,6],[461,531,5],[532,620,7],[621,729,6],[730,770,2],[771,960,3]],
               [[301,378,1],[379,430,5],[431,530,7],[531,570,6],[571,601,2],[602,740,3]],
               [[255,498,1],[499,600,2],[601,770,3]],
               [[301,544,1], [545, 672, 2], [673, 800, 3]],
               [[408, 537, 1], [538, 608, 5], [609, 794, 7], [795, 863, 6], [864, 901, 2], [902, 1040 ,3]],
               [[317, 586, 1], [587, 685, 5], [686, 737, 7], [738, 766, 6], [767, 808, 2], [809, 930, 3]],
               [[393, 662, 1], [663, 688, 5], [689, 710, 4], [711, 744, 6], [745, 1519, 1], [1520, 1595, 2], [1596, 1661, 6], [1662, 1730, 1], [1731, 1769, 5], [1770, 1836, 4], [1840, 1886, 6], [1887, 2645, 1], [2646, 2698, 5], [2699, 2958, 8], [2959, 3035, 6], [3036, 3156, 1], [3157, 3237, 5], [3238, 3416, 8], [3417, 3573, 6], [3574, 3614, 2], [3615, 3745, 6], [3746, 3795, 5], [3796, 4042, 4], [4043, 4105, 6], [4106, 4204, 1], [4205, 4264, 5], [4265, 4440, 7], [4441, 4527, 6], [4528, 5200, 1]],
               [[350, 974, 1], [975, 1315, 1], [1316, 1351, 5], [1352, 1414, 4], [1415, 1450, 6], [1451, 1750, 1], [1751, 1805, 5], [1806, 1844, 4], [1845, 4884, 6], [1885, 2490, 1], [2491, 2514, 5], [2515, 2563, 4], [2564, 2587, 6], [2588, 3040, 1], [3041, 3077, 5], [3078, 3125, 6], [3126, 3243, 1], [3244, 3353, 1], [3354, 3401, 5], [3402, 3500, 4]]])

# Synch data of 8 cameras of each senario
synch_list = [[3,3,8,4,23,6,6,0],               # senario 1
              [25,40,0,16,18,33,33,6],          # senario 2
              [12,16,8,16,35,20,20,0],          # senario 3
              [72,79,78,0,68,82,83,56],         # senario 4
              [17,24,5,11,18,26,28,0],          # senario 5
              [0,100,106,90,89,103,104,89],     # senario 6
              [28,14,16,0,1,17,18,20],          # senario 7
              [92,79,0,81,64,81,82,56],         # senario 8
              [18,9,1,19,13,11,12,0],           # senario 9
              [14,15,19,33,12,17,19,0],         # senario 10
              [23,4,20,14,0,6,7,12],            # senario 11
              [21,6,13,8,0,3,7,0],              # senario 12
              [16,33,0,7,27,27,36,13],          # senario 13
              [49,36,38,0,29,29,7,14],          # senario 14
              [15,19,19,15,34,40,23,0],         # senario 15
              [23,29,0,2,12,9,3,3],             # senario 16
              [21,26,15,0,10,0,29,18],          # senario 17
              [99,105,86,0,84,108,109,77],      # senario 18
              [19,27,16,19,5,29,0,20],          # senario 19
              [25,9,3,10,10,4,5,0],             # senario 20
              [20,30,22,3,8,33,32,0],           # senario 21
              [0,46,51,41,53,46,47,34],         # senario 22
              [31,52,52,45,54,60,50,0],         # senario 23
              [3,36,7,0,37,10,33,1]]            # senario 24

# Frame information of fall state
start2endFrame = [[874,1285],                   # senario 1
                 [308, 600],                    # senario 2
                 [380,784],                     # senario 3
                 [230,780],                     # senario 4
                 [288,450],                     # senario 5
                 [325,750],                     # senario 6
                 [330,680],                     # senario 7
                 [114,380],                     # senario 8
                 [310,760],                     # senario 9
                 [315,680],                     # senario 10
                 [378,600],                     # senario 11
                 [355,770],                     # senario 12
                 [301,960],                     # senario 13
                 [372,1115],                    # senario 14
                 [363,870],                     # senario 15
                 [380,1000],                    # senario 16
                 [251,860],                     # senario 17
                 [301,740],                     # senario 18
                 [255,770],                     # senario 19
                 [301,800],                     # senario 20
                 [408,1040],                    # senario 21
                 [317,930],                     # senario 22
                 [393,5200],                    # senario 23
                 [350,3500]]                    # senario 24
#####################################################################################################################


# Match camera synch (for dataset 1)
def match_synch(folder_num, cam_num, start, end, synch=0):
    inputImg_path = []

    cam_folder = folder_path + str(folder_num).zfill(2) + "/cam" + str(cam_num)
    print("Load cam",cam_num," in chute",str(folder_num).zfill(2))

    # Take imgae file name and sort the files
    files = os.listdir(cam_folder)
    files = natsort.natsorted(files)

    states = state_frame[folder_num - 1]

    for i in range(len(files)):
        file_path = cam_folder+"/" + files[i]
        # if os.path.isfile(file_path) and i > (synch - 1):             # for match synch
        if os.path.isfile(file_path) and (start+synch) <= i+1 and i+1 <= end+synch:   # between start frame and end frame
            for state_val in states:
                if state_val[0]+synch <= i+1 and i+1 <= state_val[1]+synch:
                    inputImg_path.append((file_path, state_val[2]))
                    print("Load complete : ",file_path,"state value : ",state_val[2])
                    break
    
    return inputImg_path

# Extract keypoints and map with state value in images(img_pathes)
def extract_pose(img_pathes, AInetwork):  
    '''
    frame 순서대로 저장된 keypoints 값 
    [[frame1의 keypoints],[frame2의 keypoints],
                  ...,
    [frame(n-1)의 keypoints],[frame(n)의 keypoints]]
    '''
    net = AInetwork 
    keypoints_list = []

    for img_path in img_pathes:
        # img 불러오기, img 처리
        image = cv2.imread(img_path[0])
        image = cv2.resize(image, dsize=(368,368), interpolation = cv2.INTER_AREA)
        imageHeight, imageWidth, _ = image.shape
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

        # network에 img 넣어주기
        net.setInput(inpBlob)

        # 결과 받아오기
        output = net.forward()

        # output.shape[0] = 이미지 ID, [2] = 출력 맵의 높이, [3] = 너비
        H = output.shape[2]
        W = output.shape[3]

        # print Debuging
        debug = img_path[0].split("/")
        debug = "/".join(debug[-3::])
        print("img<{}>".format(debug))
        print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

        lis = []
        # Insert position information of 15 parts
        for i in range(0,15):
            # 해당 신체부위 신뢰도 얻음.
            probMap = output[0, i, :, :]
        
            # global 최대값 찾기
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > 0.1:
                lis.append(point[0]/W)
                lis.append(point[1]/H)
            else :
                lis.append(None)
                lis.append(None)

        # Insert state value at last
        lis.append(img_path[1])

        keypoints_list.append(lis)

    return keypoints_list

# Save dataset as csv file
def arr2csv(data_list, folder_num, cam_num):
    s_path = save_path + "chute" + str(folder_num).zfill(2) + "/"

    # Change list to numpy array and save as .scv file
    datas = np.array(data_list)
    # Debug
    print("array's shape : ",datas.shape,", array's dtype : ",datas.dtype)

    # Make directory if doesn't exist
    if not os.path.exists(s_path):
        print("make directory : ", s_path)
        os.makedirs(s_path)

    savefile_path = s_path + "cam" + str(cam_num) + ".csv"
    df = pd.DataFrame(datas)
    ## 데이터셋 1
    # df.to_csv(savefile_path,index=False)
    ## 데이터셋 2
    df.to_csv("dataset2.csv",index=False)
    # Debuging
    print("Save succese : ",savefile_path)


def re():
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    a= (5,)   # 3, 6, 9 folder number
    for n in a:
        start = start2endFrame[n][0]
        end = start2endFrame[n][1]
        for c in range(num_of_cam):
            img = match_synch(n+1,c+1, start, end,synch_list[n][c])
            data = extract_pose(img, net)
            print("Extract complete : chute{0:02d}/cam{1}".format(n+1,c+1))
            arr2csv(data, n+1, c+1)


if __name__ == "__main__":
    # 위의 path에 있는 network 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # ## 첫번째 데이터 셋 (synch가 필요한 )
    # for n in range(num_of_folder):
    #     start = start2endFrame[n][0]
    #     end = start2endFrame[n][1]
    #     for c in range(num_of_cam):
    #         img = match_synch(n+1,c+1, start, end,synch_list[n][c])
    #         data = extract_pose(img, net)
    #         print("Extract complete : chute{0:02d}/cam{1}".format(n+1,c+1))
    #         arr2csv(data, n+1, c+1)

    ## 두번째 데이터 셋
    csv_data = pd.read_csv("/home/sejin/workspace/graduate-thesis/dataset/dataset/database_2/video/urfall-cam0-falls.csv")
    data = np.array(csv_data)

    imgs = []
    for d in data:
        imgs.append(("../dataset/fall_video/"+d[0]+"-cam0-rgb/"+d[0]+"-cam0-rgb-"+str(d[1]).zfill(3)+".png", d[2]))

    print(imgs[0])
    data = extract_pose(imgs, net)
    print("Extract complete : file{0:02d}/frame{1}".format(12,34))
    arr2csv(data, 1, 1)