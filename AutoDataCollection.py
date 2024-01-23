import csv
import copy
import argparse

import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from models import KeyPointClassifier
from utils.functions import calc_bounding_rect, calc_landmark_list, draw_bounding_rect, draw_info, draw_info_text, draw_landmarks, pre_process_landmark, select_mode, write_in_csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_load', action='store_true')
    parser.add_argument("--min_detection_confidence", help= 'min_detection_confidence', type= float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type= int, default=0.5)

    args = parser.parse_args()

    return args

def DataCollection():
    #  Argument Parsing --------------------------------------------------------------------------------------------
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_load
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    #   Camera preparation -----------------------------------------------------------------------------------------

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #   Loading the models -----------------------------------------------------------------------------------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode = use_static_image_mode,
        max_num_hands = 1, 
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
#   Read Labels ------------------------------------------------------------------------------------------------
    with open('datasets\keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]        
    #    FPS Measurement -------------------------------------------------------------------------------------------
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #    Implementation 
    mode = 1
    while True:
        fps = cvFpsCalc.get()

        #process key (esc == 27: end)
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image,1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #Sometimes I have no idea how things work but they do somehow
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounded Box Calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark Calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative co-ordinates / normalized co-ordinates 
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                #writing to the dataset
                write_in_csv(number, mode, pre_processed_landmark_list)
                # Hand Sign Classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                #Drawing on the cam
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        else:
            pass

        debug_image = draw_info(debug_image, fps, number, mode)

        # Screen Name
        cv.imshow('Data Collection Assistance Application', debug_image)
    cap.release()
    cv.destroyAllWindows()

DataCollection()