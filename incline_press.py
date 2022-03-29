import math
import cv2
import mediapipe as mp
import time
import _thread
import csv
import os
from itertools import count

from util import *

COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)

LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

reps = 0
good_reps = 0

# instantiate pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# mp_draw = mp.solutions.drawing_utils
# mp_styles = mp.solutions.drawing_styles

# start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ok, img = cap.read()
    
    # detect pose
    img.flags.writeable = False # done temporarily to improve performance
    current_pose = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img.flags.writeable = True

    # analyze pose
    if current_pose.pose_landmarks:
        
        landmarks = []
        for id, landmark in enumerate(current_pose.pose_landmarks.landmark):
            h, w, _ = img.shape
            pixel_x, pixel_y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([id, pixel_x, pixel_y, landmark.visibility])

        if len(landmarks) > 0:

            # find relevant points of whichever side is more visible
            left_vis = landmarks[LEFT_ELBOW][3]
            right_vis = landmarks[RIGHT_ELBOW][3]

            ear, shoulder, elbow, wrist, hip, knee = RIGHT_EAR, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE

            if left_vis > right_vis:
                ear, shoulder, elbow, wrist, hip, knee = LEFT_EAR, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE

            ear_point = find_point_position(ear, landmarks)
            shoulder_point = find_point_position(shoulder, landmarks)
            elbow_point = find_point_position(elbow, landmarks)
            wrist_point = find_point_position(wrist, landmarks)
            hip_point = find_point_position(hip, landmarks)
            knee_point = find_midpoint(LEFT_KNEE, RIGHT_KNEE, landmarks)

            # calculate angle of incline from right shoulder, right hip, and knees
            
            incline_points = (shoulder_point, hip_point, knee_point)
            incline_angle = 180 - calculate_angle(incline_points)

            # check and plot incline angle (expert advice)
            incline_color = COLOR_RED

            if incline_angle <= 48 and incline_angle >= 27:
                incline_color = COLOR_GREEN
            elif incline_angle <= 55 and incline_angle >= 20:
                incline_color = COLOR_YELLOW

            torso_line = get_line_segment(shoulder, hip, landmarks)
            thigh_line = get_line_segment(hip, knee, landmarks)
            plot_line(torso_line[0], torso_line[1], incline_color, img)
            plot_line(thigh_line[0], thigh_line[1], incline_color, img)
            plot_label(torso_line[1], incline_angle, COLOR_WHITE, img)

            # check and plot upper arm angle (expert advice)
            elbow_shoulder_dx = abs(elbow_point[0] - shoulder_point[0])
            torso_dx = abs(shoulder_point[0] - hip_point[0])

            # if elbow is below shoulder (i.e., the beginning/end of the rep)
            if elbow_point[1] > shoulder_point[1]:

                # if elbow.x is between shoulder.x and hips.x
                if ((shoulder_point[0] < elbow_point[0] and elbow_point[0] < hip_point[0])
                or (hip_point[0] < elbow_point[0] and elbow_point[0] < shoulder_point[0])):

                    if elbow_shoulder_dx >= 0.15 * torso_dx and elbow_shoulder_dx <= 0.25 * torso_dx:
                        upper_arm_color = COLOR_GREEN
                    elif elbow_shoulder_dx >= 0.1 * torso_dx and elbow_shoulder_dx <= 0.3 * torso_dx:
                        upper_arm_color = COLOR_YELLOW
                    else:
                        upper_arm_color = COLOR_RED

                else:
                    upper_arm_color = COLOR_RED
            else:
                upper_arm_color = COLOR_WHITE

            upper_arm_line = get_line_segment(shoulder, elbow, landmarks)
            plot_line(upper_arm_line[0], upper_arm_line[1], upper_arm_color, img)

            # check and plot full arm angle (expert advice)
            arm_extension_points = (shoulder_point, elbow_point, wrist_point)
            arm_extension_angle = calculate_angle(arm_extension_points)

            # might not work with actual weights; might need to use arUco marker instead
            arm_extended = arm_extension_angle > 170 and arm_extension_angle < 180 and wrist_point[1] < ear_point[1]
            

    # show final image
    cv2.imshow("AI Gym Assistant", img)
    
    # quit with 'q' key
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()