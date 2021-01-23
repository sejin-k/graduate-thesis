from enum import Enum


class Actions(Enum):
    """
    Actions enum
    """
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3

    # # framewise_recognition_under_scene.h5
    # stand = 0
    # walk = 1
    # operate = 2
    # fall_down = 3
    # # run = 4

    # # fall_detection_v2.h5
    # moving = 0
    # fall_down = 1

    # fall_detection_v9.h5
    fall_down = 0
    moveing__ = 1
    moveing = 2