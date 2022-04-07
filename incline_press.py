import math
import cv2
import mediapipe as mp
import time
import _thread
import csv
import os
from itertools import count

from util import *

def check_incline_press():

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

                # find positions of relevant points
                ear_point = find_point_position(ear, landmarks)
                shoulder_point = find_point_position(shoulder, landmarks)
                elbow_point = find_point_position(elbow, landmarks)
                wrist_point = find_point_position(wrist, landmarks)
                hip_point = find_point_position(hip, landmarks)
                knee_point = find_midpoint(LEFT_KNEE, RIGHT_KNEE, landmarks)

                # calculate angle of incline from right shoulder, right hip, and knees
                incline_points = (shoulder_point, hip_point, knee_point)
                incline_angle = 180 - calculate_angle(incline_points)
                incline_color = classify_incline_angle(incline_angle)

                torso_line = get_line_segment(shoulder, hip, landmarks)
                thigh_line = get_line_segment(hip, knee, landmarks)

                # plots green, yellow, or red depending on angle measure, along with the actual angle
                plot_line(torso_line[0], torso_line[1], incline_color, img)
                plot_line(thigh_line[0], thigh_line[1], incline_color, img)
                plot_label(torso_line[1], incline_angle, COLOR_WHITE, img)

                # check and plot upper arm angle (expert advice)
                elbow_shoulder_dx = abs(elbow_point[0] - shoulder_point[0])
                torso_dx = abs(shoulder_point[0] - hip_point[0])

                upper_arm_color = classify_upper_arm(shoulder_point, elbow_point, hip_point, elbow_shoulder_dx, torso_dx)

                upper_arm_line = get_line_segment(shoulder, elbow, landmarks)
                plot_line(upper_arm_line[0], upper_arm_line[1], upper_arm_color, img)

                # check and plot full arm angle (expert advice)
                arm_extension_points = (shoulder_point, elbow_point, wrist_point)
                arm_extension_angle = calculate_angle(arm_extension_points)

                # might not work with actual weights; might need to use arUco marker instead of wrist
                arm_extended = arm_extension_angle > 160 and arm_extension_angle < 180 and wrist_point[1] < ear_point[1]
                
                if arm_extended:
                    plot_rectangle(img, 'Arm Extension', (32, 32), (96, 32), (128, 96), COLOR_GREEN)

        # show final image
        cv2.imshow("AI Gym Assistant", img)
        
        # quit with 'q' key
        if cv2.waitKey(5) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
Parameters:
    incline_angle: the user's angle of incline, taken from the shoulder, hip, and midpoint between the knees.
Determines if the user's angle of incline is good, OK, or poor.
Returns a color to be plotted on the image.
'''
def classify_incline_angle(incline_angle):

    # check and plot incline angle (expert advice)
    incline_color = COLOR_RED

    if incline_angle <= 48 and incline_angle >= 27:
        incline_color = COLOR_GREEN
    elif incline_angle <= 55 and incline_angle >= 20:
        incline_color = COLOR_YELLOW

    return incline_color

'''
Parameters:
    shoulder_point, elbow_point, hip_point: 2D points of the respective body parts
    elbow_shoulder_dx: the x-distance between the user's elbow and shoulder
    torso_dx: the x-distance between the user's shoulder and hip
First determines if the upper arm is in the proper place to be tracked, then determines if angle is
good, OK, or poor.
Returns a color to be plotted on the image.
'''
def classify_upper_arm(shoulder_point, elbow_point, hip_point, elbow_shoulder_dx, torso_dx):

    upper_arm_color = COLOR_WHITE

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

    return upper_arm_color

if __name__ == '__main__':
    check_incline_press()