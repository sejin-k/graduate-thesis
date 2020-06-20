import cv2 as cv
import pathlib
import os

root_path = '../dataset/videos/'
save_path = '../dataset/frames/'
folder_name = 'chute'
video_extension = '.avi'
num_folder = 24
num_cam = 8

def make_path (openOrsave, num_f, num_c):
    # save = 1, open = 0
    if openOrsave == 0:
        path = root_path + folder_name + str(num_f).zfill(2) + '/cam' + str(num_c) + video_extension
    elif openOrsave == 1:
        path = save_path + folder_name + str(num_f).zfill(2) + '/cam' + str(num_c) +'/'
    else:
        print('folder or cma number is Out of range')

    return path

def make_folder():
    for n in range(num_folder):
        for c_n in range(num_cam):

            s_path = make_path(1, n+1, c_n+1)
            # 디렉터리 없을경우 dir 생성
            if not os.path.exists(s_path):
                os.makedirs(s_path)

def video2frames ():
    for n in range(num_folder):
        for c_n in range(num_cam):
            v_path = make_path(0, n+1, c_n+1)
            # print(v_path)

            # 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
            vidcap = cv.VideoCapture(v_path)
            
            count = 0
            # 저장할 경로
            s_path = make_path(1, n+1, c_n+1)

            # # 디렉터리 없을경우 dir 생성
            # if not os.path.exists(s_path):
            #     os.makedirs(s_path)

            try:
                while(vidcap.isOpened()):
                    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
                    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
                    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
                    ret, image = vidcap.read()
                
                    # 캡쳐된 이미지를 저장하는 함수
                    cv.imwrite(s_path + "frame%d.jpg" % count, image)
                
                    print('Saved ' + s_path + 'frame%d.jpg' % count)
                    count += 1
                
                vidcap.release()
            except cv.error as e:
                print(e)
                pass

if __name__ == "__main__":
    # make_folder()
    video2frames()