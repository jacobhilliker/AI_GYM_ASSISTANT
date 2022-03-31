# importing modules
# !pip install opencv-python mediapipe

import math
import cv2
import mediapipe as mp
import time
import _thread
import csv
import os
from itertools import count

from Shoulder_version_1_function_module import *

# Initiation
index = count()
ptime = 0
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)
color_black = (0,0,0)
color_white = (255,255,255)
good_count = 0
direction = 1
count = 0
point_no = []


# Openpose module
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Tracking the detected marker
tracker = cv2.TrackerCSRT_create()

# Capture the video feed
cap = cv2.VideoCapture(0)

# Run the code for plotting 
_thread.start_new_thread(graph_plot, ())

# Creating a CSV file
num_coord = 33
landmarks = ["Point_no", "B_X0", "B_Y0"]
for val in range(1, num_coord + 1):
    landmarks += [f'x{val}', f'y{val}']

with  open('aruko_marker.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

# Initial_Run for detecting the marker
initial_run_count = 10

while initial_run_count > 0:
    ok, img = cap.read()
    arucofound = findArucoMarkers(img)

    if len(arucofound[0]) != 0:
        bounding_box = plot_ArucoMarkers(arucofound, img)
        initial_run_count -= 1

    cv2.imshow("Tracking", img)
    cv2.waitKey(30)
cv2.destroyAllWindows()

# Detecting and tracking the marker
while cap.isOpened():
    try:
        ok = tracker.init(img, bounding_box)
    except:
        pass

    ok, img = cap.read()
    

    timer = cv2.getTickCount()

    arucofound = findArucoMarkers(img)

    if len(arucofound[0]) != 0:
        bounding_box = plot_ArucoMarkers(arucofound, img)
    else:
        try:
            ok, bounding_box = tracker.update(img)
        except:
            pass

    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # print(fps)

    # Draw bounding box
    if ok:

        if (int(bounding_box[0]) + int(bounding_box[2])) == int(bounding_box[0]) or (
                int(bounding_box[1]) + int(bounding_box[3])) == int(bounding_box[1]):
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[2]), int(bounding_box[3]))
        else:
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))

        centroid_tracking = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        cv2.circle(img, (centroid_tracking[0], centroid_tracking[1]), 3, (255, 0, 0), 3)

    # Pose Detection
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:

        lmlist = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])

        if len(lmlist) != 0:
            # Calculate left arm
            # find left arm
            point_leftarm = findpositions(23, 11, 13, lmlist)
            angle_leftarm = calculate_angle(point_leftarm)
            # color left arm (NEED EXPERT ADVICE)
            if angle_leftarm >= 90 and angle_leftarm < 180:
                color_leftarm = color_green
            elif angle_leftarm > 70 and angle_leftarm < 90:
                color_leftarm = color_yellow
            else:
                color_leftarm = color_red
            # plot left arm
            cv2.putText(img, str('Left Arm'), (520, 90),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 75), (625, 100), color_leftarm, cv2.FILLED)
            plot1 = plot(point_leftarm, color_leftarm, angle_leftarm, img)

            # Calculate left forearm
            # find left forearm
            leftforearm_line = get_line_segment(13, 15, lmlist)
            # calculate slope of arm (slope should be very large)
            rise = leftforearm_line[1][1] - leftforearm_line[0][1]
            run = leftforearm_line[1][0] - leftforearm_line[0][0]
            if run == 0:
                run = 0.0001
            slope = rise/run
            slope = abs(round(slope))
            # angle limits (NEED EXPERT ADVICE)
            if slope > 6:
                color_leftforearm = color_green
            elif slope > 3:
                color_leftforearm = color_yellow
            else:
                color_leftforearm = color_red
            # plot left forearm
            plot_leftforearm = plot_line(leftforearm_line[0], leftforearm_line[1], color_leftforearm, img)
            leftforearm_midpoint = get_midpoint(leftforearm_line)
            cv2.putText(img, str('Left Forearm'), (485, 40),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 25), (625, 50), color_leftforearm, cv2.FILLED)
            
            # Calculate right arm
            # find right arm
            point_rightarm = findpositions(24, 12, 14, lmlist)
            angle_rightarm = calculate_angle(point_rightarm)
            # color right arm (NEED EXPERT ADVICE)
            if angle_rightarm >= 90 and angle_rightarm < 180:
                color_rightarm = color_green
            elif angle_rightarm > 70 and angle_rightarm < 90:
                color_rightarm = color_yellow
            else:
                color_rightarm = color_red
            # plot left arm
            cv2.putText(img, str('Right Arm'), (515, 190),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 175), (625, 200), color_rightarm, cv2.FILLED)
            plot2 = plot(point_rightarm, color_rightarm, angle_rightarm, img)

            # Calculate right forearm
            # find right forearm
            rightforearm_line = get_line_segment(14, 16, lmlist)
            # calculate slope of arm (slope should be very large)
            rise1 = rightforearm_line[1][1] - rightforearm_line[0][1]
            run1 = rightforearm_line[1][0] - rightforearm_line[0][0]
            if run1 == 0:
                run1 = 0.0001
            slope1 = rise1/run1
            slope1 = abs(round(slope1))
            # angle limits (NEED EXPERT ADVICE)
            if slope1 > 6:
                color_rightforearm = color_green
            elif slope1 > 3:
                color_rightforearm = color_yellow
            else:
                color_rightforearm = color_red
            # plot right forearm
            plot_rightforearm = plot_line(rightforearm_line[0], rightforearm_line[1], color_rightforearm, img)
            rightforearm_midpoint = get_midpoint(rightforearm_line)
            cv2.putText(img, str('Right Forearm'), (480, 140),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (600, 125), (625, 150), color_rightforearm, cv2.FILLED)

            # Calculate reps
            # grab left hand and left shoulder points
            point_lefthand = find_point_position(15, lmlist)
            point_leftshoulder = find_point_position(11, lmlist)
            # track distance from left hand to left shoulder
            lefthand_leftshoulder_distance = abs(point_lefthand[1] - point_leftshoulder[1])
            # grab right hand and right shoulder points
            point_righthand = find_point_position(16, lmlist)
            point_rightshoulder = find_point_position(12, lmlist)
            # track distance from right hand to right shoulder
            rightthand_righttshoulder_distance = abs(point_righthand[1] - point_rightshoulder[1])
            # placeholders
            benchMin = 50
            benchMax = 300
            benchMinMax = (benchMin, benchMax)
            # plot left arm
            plot3 = plot_bar_distance(lefthand_leftshoulder_distance, benchMinMax, img)
            # plot right arm
            plot4 = plot_bar_distance(rightthand_righttshoulder_distance, benchMinMax, img)
            # color list
            color_list = [color_leftforearm, color_leftarm, color_rightforearm, color_rightarm]
            # rep count
            if plot3[0] == 100 and plot4[0] == 100:
                if direction == 0:
                    direction = 1
                    count += 0.5
                    if color_red not in color_list:
                        good_count += 0.5
                    else:
                        good_count += 0
            if plot3[0] == 0 and plot4[0] == 0:
                if direction == 1:
                    direction = 0
                    count += 0.5
                    if color_red not in color_list:
                        good_count += 0.5
                    else:
                        good_count += 0
            # plot reps
            pose1 = results.pose_landmarks.landmark
            pose_data = list(
                np.array([[int((landmark.x) * w), int((landmark.y) * h)] for landmark in pose1]).flatten())
            dumbel_data = list(np.array([centroid_tracking[0], centroid_tracking[1]]))
            point_no = list(np.array([next(index)]))

            combined_data = point_no + dumbel_data + pose_data

            with  open('aruko_marker.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(combined_data)

            cv2.putText(img, 'Total_REPS', (25, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 5), (170, 35), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (130, 35),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, 'Good_REPS', (25, 75),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (120, 50), (170, 80), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(good_count)), (130, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            
    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()