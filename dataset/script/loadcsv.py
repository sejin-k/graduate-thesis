import pandas as pd
import numpy as np

dataRange = 16  # Under and over frame range
threshold_zero = 10

# Fins fall state frame number in csv data file
def find_fallFrame(csv_path):
    '''
    Find fall state frame in .csv  file
    Arg :
        csv_path (string) : csv file's path
    Return :
        data (np.array) : Data stored in the csv file converted to numpy array
        startNendFrame (list) : Fall start and fall end frame number in tuple
    '''
    startNendFrame = []
    startFrame, endFrame = 0, 0
    frame_number = 1
    fall = False
    save = True

    # load csv file
    csv_data = pd.read_csv(csv_path)
    data = np.array(csv_data)

    # Find start and end frame number
    for d in data:
        if d[-1] == 2 and  fall == False and save == True:
            startFrame = frame_number
            fall = True
            save = False
        if d[-1] != 2 and fall == True and save == False:
            endFrame = frame_number
            if save == False:
                startNendFrame.append((startFrame, endFrame))
                save = True
                fall = False
        
        frame_number += 1

    # Debug path, start and end frame
    print("# Data path : {}".format(csv_path))
    for info in startNendFrame:
        print("# Start frame number : {}, # End frame number : {}".format(info[0], info[1]))

    return (data, startNendFrame)

# Extract sequence data to use
def extract_seq_data(data, frames):
    '''
    Make train data from data
    Arg :
        data (np.array) : Fall state data array
        frames (list) : Have tuple element [0] is start frame number [1] is end frame numper => (start frame number, end frame numper) 
    Return :
        seq_data (np.array) : Dataset for model training
    '''
    seq_data = []
    i = 0
    # count = 0
    
    for frame_info in frames:
        # Just take (start - dataRange)frame ~ (end + dataRange)frame
        for i in range(frame_info[0]-dataRange, frame_info[1]+dataRange+1):
            seq_data.append(data[i])

        # # Exclude data that have not head and neck information -> so many data loss head and neck data
        # # dataRange ~ startFrame
        # while(count <= dataRange):
        #     if (data[frame_info[0] - i+1][0] !=0 and data[frame_info[0] - i+1][1] !=0 and data[frame_info[0] - i+1][2] !=0 and data[frame_info[0] - i+1][3] !=0):
        #         seq_data.append(np.array((data[frame_info[0] - i+1][0], data[frame_info[0] - i+1][1], data[frame_info[0] - i+1][2], data[frame_info[0] - i+1][3], data[frame_info[0] - i+1][-1])))
        #         count += 1
        #     i += 1
        # count, i = 0, 0

        # # startFrame ~ endFrame
        # for j in range(frame_info[0], frame_info[1]+1):
        #     seq_data.append(np.array((data[j][0], data[j][1], data[j][2], data[j][3], data[j][-1])))
        
        # # endFrame ~ dataRange
        # while(count <= dataRange):
        #     if (data[frame_info[1] + i+1][0] !=0 and data[frame_info[1] + i+1][1] !=0 and data[frame_info[1] + i+1][2] !=0 and data[frame_info[1] + i+1][3] !=0):
        #         seq_data.append(np.array((data[frame_info[1] + i+1][0], data[frame_info[1] + i+1][1], data[frame_info[1] + i+1][2], data[frame_info[1] + i+1][3], data[frame_info[1] + i+1][-1])))
        #         count += 1
        #     i += 1
        # count, i = 0, 0        

    seq_data = np.array(seq_data)
    return seq_data

# scv file -> find_fallFrame -> extract_seq_data -> seq data
def fall_seq_data(data_path):
    '''
    Make data for fall detection rnn model
    <using find_fallFrame, extract_seq_data function>
    Arg:
        data_path (string) : File path of .csv file
    Return :
        result (np.array) : Falling Data From Before dataRange Frames to After dataRange Frames. shape is (None, 31)
    '''
    data, framepoints = find_fallFrame(data_path)
    result = extract_seq_data(data, framepoints)

    return result

# Chage sequence data (N,31) to train data (N, 10 , 31)
def train_data(seqData, seq_size = 10):
    dataset = []
    subset = []
    # zero_num = 0
    # count = 0

    # for i in range(len(seqData)):
        # ### 0이 많이 포함된 데이터 제거 (zero_num < threshold_zero)
        # for j in seqData[i]:
        #     if j == 0:
        #         zero_num += 1
        # if zero_num < threshold_zero:
        #     subset.append(np.array(seqData[i]))
        #     count += 1
        # zero_num = 0
        # if count == seq_size:
        #     dataset.append(np.array(subset))
        #     subset = []
        #     count = 0

        # ### head와 neck만 있는 데이터
        # if seqData[i][0] != 0 and seqData[i][1] != 0 and seqData[i][2] != 0 and seqData[i][3] != 0:      # head x,y / neck x,y
        #     # print(seqData[i][0], seqData[i][1], seqData[i][2], seqData[i][3], seqData[i][-1])
        #     subset.append(np.array((seqData[i][0], seqData[i][1], seqData[i][2], seqData[i][3], seqData[i][-1])))
        #     count += 1
        #     if count == seq_size:
        #         dataset.append(np.array(subset))
        #         subset = []
        #         count = 0

    # Too many data have 0
    for i in range(len(seqData) - seq_size):
        subset = seqData[i : (i + seq_size)]
        dataset.append(subset)
    
    return np.array(dataset)


# For Test
if __name__ == "__main__":
    a, b = find_fallFrame('../dataset/dataset/csvfile/chute23/cam1.csv')
    # print(a, b)
    c = extract_seq_data(a, b)
    print(c)
    print(c.shape)
    d = train_data(c)
    print(d)
    print(d.shape)
    print(d[0])
    # seq_d = fall_seq_data('../dataset/dataset/csvfile/chute23/cam1.csv')
    # t_d = train_data(seq_d)
    # print(t_d)
    # print(t_d.shape)