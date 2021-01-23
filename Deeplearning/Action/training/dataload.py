import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loadcsv import fall_seq_data, train_data

dataset_path = "/home/sejin/workspace/graduate-thesis/dataset/dataset/database_1/csvfile/"
folder_name = "chute"
number_of_folder = 23
cmera_name = "cam"
number_of_cam = 8

sequence_num = 10

dataset = []

def make_train_dataset(folderNum, camNum):
    dataPath = dataset_path + folder_name + str(folderNum).zfill(2)+ "/" + cmera_name + str(camNum) + ".csv"
    seqData = fall_seq_data(dataPath)
    trainData = train_data(seqData, sequence_num)
    print(trainData.shape)
    for data in trainData:
        dataset.append(data)


def fall_detection_dataset(folder_num=number_of_folder, cam_num=number_of_cam):
    for f in range(folder_num):
        for c in range(cam_num):
            make_train_dataset(f+1, c+1)
    
    fall_dataset = np.array(dataset)
    print("Shape of dataset : {}".format(fall_dataset.shape))

    ## 넘어짐 상태가 아닌것은 0 넘어짐 상태인 것은 1 ##
    for f_d in fall_dataset:
        for d in f_d:
            if d[-1] != 2:
                d[-1] = 0
            else:
                d[-1] = 1

    return fall_dataset


#### nan, zero 데이터 처리하는 함수
def fill_blank_zero(folderNum, camNum):
    # dataPath = dataset_path + folder_name + str(folderNum).zfill(2)+ "/" + cmera_name + str(camNum) + ".csv"
    # csv_data = pd.read_csv(dataPath)
    # data = np.array(csv_data)
    # print(data.shape)
    csv_data = pd.read_csv('dataset_2_3state.csv')
    data = np.array(csv_data)
    print(data.shape)
    
    for d in data:
        for i in range(len(d)):
            if np.isnan(d[i]):
                   d[i] = 0

    data = np.round_(data,3)
    df = pd.DataFrame(data)
    df.to_csv("dataset_2_3state_zero.csv",index=False)


def fill_blank_mean(folderNum, camNum):
    np.set_printoptions(threshold=np.nan)

    dataPath = dataset_path + folder_name + str(folderNum).zfill(2)+ "/" + cmera_name + str(camNum) + ".csv"
    csv_data = pd.read_csv(dataPath)
    data = np.array(csv_data)
    print(data.shape)
    X = data[:,:-1]
    # print(X)


def one_y_data(fall_dataset):
    no, yes = 0, 0

    y = fall_dataset[:,:,-1]
    new_state = np.zeros((y.shape[0],1))
    for state in range(len(y)):
        for s in y[state]:
            if s == 0:
                no += 1
            if s== 1:
                yes+= 1

        if yes >= 5:
            new_state[state] = 1
        else :
            new_state[state] = 0
        no ,yes =0,0

    print(new_state)
    return new_state




if __name__ == "__main__":
    # # Fill blank in nan
    # for f in range(number_of_folder):
    #     for c in range(number_of_cam):
    #         fill_blank_zero(f+1,c+1)
    # pass

    # make_train_dataset(1,1)
    # print(dataset)
    # print(np.array(dataset).shape)

    # a = fall_detection_dataset(22,8)
    # print(a)
    # print(a.shape)

    # e = one_y_data(np.array(dataset))
    # print(e)
    # print(e.shape)
    
    fill_blank_zero(0,0)
    # a = fall_detection_dataset(22,8)

    # print(a.shape)