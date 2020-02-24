import cv2
import numpy as np
import argparse
import pickle
import glob
import random

class Display:

    def __init__(self, x_lim, y_lim, sp=30):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.sp = sp

        # font settig
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (int(self.x_lim/54), int(self.y_lim/2))
        self.topLeftConnerOfText = (int(self.x_lim/54), int(self.y_lim/19))
        self.fontScale = 0.4
        self.fontColor = (255,255,255)
        self.lineType = 1
            
    def draw_frame(self, landmark, is_center, text=None, title=None):
        frame = np.zeros((self.x_lim, self.y_lim, 3), np.uint8)
        
        left_eye_region = np.array(list(zip(landmark[4:16:2], landmark[5:16:2])), np.int32)
        right_eye_region = np.array(list(zip(landmark[16:28:2], landmark[17:28:2])), np.int32)
        # draw on frame
        cv2.polylines(frame, [left_eye_region], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_eye_region], True, (255, 255, 255), 1)

        right_pupil = np.array(landmark[0:2], np.int32)
        left_pupil = np.array(landmark[2:4], np.int32)
        cv2.circle(frame, (left_pupil[0], left_pupil[1]), 3, (255, 255, 255), -1)
        cv2.circle(frame, (right_pupil[0], right_pupil[1]), 3, (255, 255, 255), -1)

        right_eyebrow = list(zip(landmark[28:38:2], landmark[29:38:2]))
        for index, item in enumerate(right_eyebrow):
            if index == len(right_eyebrow) - 1:
                break
            cv2.line(frame, item, right_eyebrow[index + 1], (255, 255, 255), 1)
        
        left_eyebrow = list(zip(landmark[38:48:2], landmark[39:48:2]))
        for index, item in enumerate(left_eyebrow):
            if index == len(left_eyebrow) - 1:
                break
            cv2.line(frame, item, left_eyebrow[index + 1], (255, 255, 255), 1)

        if is_center:
            center_dot = (landmark[48], landmark[29])
            cv2.circle(frame, center_dot, 2, (255, 0, 0), -1)

        # put text
        if text:
            cv2.putText(frame, text, 
                        self.bottomLeftCornerOfText, 
                        self.font, 
                        self.fontScale,
                        self.fontColor,
                        self.lineType)

        # put current video text
        if title:
            cv2.putText(frame, 'Current_vid: {}'.format(title), 
                        self.topLeftConnerOfText, 
                        self.font, 
                        self.fontScale,
                        self.fontColor,
                        self.lineType)

        return frame

    def display(self, eye_motion_list):
        for eye_motion in eye_motion_list:
            frame = self.draw_frame(eye_motion, False)
            cv2.imshow('display', frame)
            if cv2.waitKey(self.sp) & 0xFF == ord('q'):
                exit(-1) 