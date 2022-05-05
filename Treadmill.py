# importing modules
# !pip install opencv-python mediapipe

import math
import cv2
import mediapipe as mp
from itertools import count

from util import *

# Initiation
index = count()
ptime = 0
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)
good_count = 0
direction = 0
count = 0
point_no = []

# Openpose module
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Tracking the detected marker
#tracker = cv2.TrackerCSRT_create()

# Capture the video feed
cap = cv2.VideoCapture('Tuoti_Riley_Demo_Trim.mp4')

# Run the code for plotting
#_thread.start_new_thread(graph_plot, ())

# Creating a CSV file
num_coord = 33
landmarks = ["Point_no", "B_X0", "B_Y0"]
for val in range(1, num_coord + 1):
    landmarks += [f'x{val}', f'y{val}']

# with  open('aruko_marker.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(landmarks)


# Initial_Run for detecting the marker
# initial_run_count = 10
#
# while initial_run_count > 0:
#     ok, img = cap.read()
#     arucofound = findArucoMarkers(img)
#
#     if len(arucofound[0]) != 0:
#         bounding_box = plot_ArucoMarkers(arucofound, img)
#         initial_run_count -= 1
#
#     cv2.imshow("Tracking", img)
#     cv2.waitKey(30)
# cv2.destroyAllWindows()

# Detecting and tracking the marker
# to-do: Add Timer
# startTimer = time.time()
# good_reps = False

ok, img = cap.read()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('testVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

while cap.isOpened():

    ok, img = cap.read()

    timer = cv2.getTickCount()

    #arucofound = findArucoMarkers(img)
    #
    # if len(arucofound[0]) != 0:
    #     bounding_box = plot_ArucoMarkers(arucofound, img)
    # else:
    #     try:
    #         ok, bounding_box = tracker.update(img)
    #     except:
    #         pass

    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # print(fps)

    # Draw bounding box
    # if ok:
    #
    #     if (int(bounding_box[0]) + int(bounding_box[2])) == int(bounding_box[0]) or (
    #             int(bounding_box[1]) + int(bounding_box[3])) == int(bounding_box[1]):
    #         p1 = (int(bounding_box[0]), int(bounding_box[1]))
    #         p2 = (int(bounding_box[2]), int(bounding_box[3]))
    #     else:
    #         p1 = (int(bounding_box[0]), int(bounding_box[1]))
    #         p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
    #
    #     centroid_tracking = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
    #
    #     cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
    #     cv2.circle(img, (centroid_tracking[0], centroid_tracking[1]), 3, (255, 0, 0), 3)

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

            # Calculate angle back
            point_back_left = find_positions(11, 23, 25, lmlist)
            angle_back_left = calculate_angle(point_back_left)

            point_back_right = find_positions(12, 24, 26, lmlist)
            angle_back_right = calculate_angle(point_back_right)

            color_back_right = color_green
            color_back_left = color_green

            # calculate stride length
            point_left_heel, point_right_heel = find_point_position(29, lmlist), find_point_position(30, lmlist)
            heel_distance = ((point_left_heel[0] - point_right_heel[0])**2 + (point_left_heel[1] - point_right_heel[1])**2)**0.5

            #print(heel_distance)

            if heel_distance > 320:
                cv2.putText(img, "Stride Length Too Far", (350, 350), cv2.FONT_HERSHEY_PLAIN, 3, color_red, 2)
                #print("STRIDE LENGTH TOO FAR!!!")

            point_arm_left = find_positions(11, 13, 15, lmlist)
            point_arm_right = find_positions(12, 14, 16, lmlist)

            angle_arm_left, angle_arm_right = calculate_angle(point_arm_left), calculate_angle(point_arm_right)
            # color arms, if > 90 degrees (plus 10 for some leeway), then red, 5 above yellow
            if angle_arm_left <= 92:
                color_arm_left = color_green
            elif angle_arm_left <= 96:
                color_arm_left = color_yellow
            else:
                color_arm_left = color_red

            if angle_arm_right <= 92:
                color_arm_right = color_green
            elif angle_arm_right <= 96:
                color_arm_right = color_yellow
            else:
                color_arm_right = color_red

            arm_left_plot = plot(point_arm_left, color_arm_left, angle_arm_left, img)
            arm_right_plot = plot(point_arm_right, color_arm_right, angle_arm_right, img)

            back_left_plot = plot(point_back_left, color_back_left, angle_back_left, img)
            back_right_plot = plot(point_back_right, color_back_right, angle_back_right, img)

            # make sure opposite arm and leg is swinging together (not out of sync)
            left_arm_centroid = find_centroid(13, 15, lmlist)
            right_arm_centroid = find_centroid(14, 16, lmlist)

            plot_point(left_arm_centroid, color_green, img)
            plot_point(right_arm_centroid, color_green, img)
            # compare it with the knee
            left_knee_point, right_knee_point = find_point_position(25, lmlist), find_point_position(26, lmlist)

            # find which arm is the "forward" arm
            left_wrist_point, right_wrist_point = find_point_position(15, lmlist), find_point_position(16, lmlist)
            left_hip, right_hip = find_point_position(23, lmlist), find_point_position(24, lmlist)
            # find horizontal abs distance to corresponding hip, greater one is the "forward arm"

            left_wrist_distance = abs(left_wrist_point[0] - left_hip[0])
            right_wrist_distance = abs(right_wrist_point[0] - right_hip[0])

            dominant_arm = 'NONE'
            if left_wrist_distance > right_wrist_distance:
                dominant_arm = 'LEFT'
            else:
                dominant_arm = 'RIGHT'

            # find which left is the "forward" leg
            left_toe_point, right_toe_point = find_point_position(31, lmlist), find_point_position(32, lmlist)
            # find horizontal abs distance to corresponding hip, greater one is the "forward leg"

            left_toe_distance = abs(left_toe_point[0] - left_hip[0])
            right_toe_distance = abs(right_toe_point[0] - right_hip[0])

            dominant_leg = 'NONE'
            if left_toe_distance > right_toe_distance:
                dominant_leg = 'LEFT'
            else:
                dominant_leg = 'RIGHT'

            # Make sure opposite arm and leg are dominant
            if dominant_arm == dominant_leg:
                cv2.putText(img, "Wrong Arm + Leg", (250, 250), cv2.FONT_HERSHEY_PLAIN, 3, color_red, 2)

            # Use dominant leg to ensure that heel hits before toe
            toe_left_test = find_positions(25, 29, 31, lmlist)
            toe_left_angle = calculate_angle(toe_left_test)

            toe_right_test = find_positions(26, 30, 32, lmlist)
            toe_right_angle = calculate_angle(toe_right_test)

            if dominant_leg == 'LEFT':
                print(toe_left_angle)
                if toe_left_angle > 95:
                    cv2.putText(img, "Toe Before Heel", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, color_red, 2)
            else:
                print(toe_right_angle)
                if toe_right_angle > 95:
                    cv2.putText(img, "Toe Before Heel", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, color_red, 2)

            # Calculate knee angle, knee position, toe_left position for left leg

            point_knee_left = find_positions(23, 25, 27, lmlist)
            angle_knee_left = calculate_angle(point_knee_left)
            knee_position_left = point_knee_left[1][0]

            point_toe_left = find_positions(25, 27, 31, lmlist)
            angle_toe_left = calculate_angle(point_toe_left)
            toe_position_left = point_knee_left[2][0]

            # Calculating knee overflow through foot distance
            ankle_left = find_point_position(27, lmlist)
            toe_left = find_point_position(31, lmlist)
            foot_length_left = int(math.sqrt((ankle_left[0] - toe_left[0]) ** 2 + (ankle_left[1] - toe_left[1]) ** 2))

            distance_knee_toe_left = abs(knee_position_left - toe_position_left)
            if distance_knee_toe_left < 1.1 * foot_length_left:  # EXPERT ADVICE
                color_knee_left = color_green
            elif distance_knee_toe_left < 1.3 * foot_length_left:  # EXPERT ADVICE
                color_knee_left = color_yellow
            else:
                color_knee_left = color_red

            # cv2.putText(img, str('Knee'), (550, 90),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.rectangle(img, (600, 75), (625, 100), color_knee_left, cv2.FILLED)
            left_leg_plot = plot(point_knee_left, color_knee_left, abs(knee_position_left - toe_position_left), img)

            # Calculate knee angle ,knee position ,toe_left position for right leg

            point_knee_right = find_positions(24, 26, 28, lmlist)
            angle_knee_right = calculate_angle(point_knee_right)
            knee_position_right = point_knee_right[1][0]

            point_toe_right = find_positions(26, 28, 32, lmlist)
            angle_toe_right = calculate_angle(point_toe_right)
            toe_position_right = point_knee_right[2][0]

            # Calculating knee overflow through foot distance
            ankle_right = find_point_position(28, lmlist)
            toe_right = find_point_position(32, lmlist)
            foot_length_right = int(math.sqrt((ankle_right[0] - toe_right[0]) ** 2 + (ankle_right[1] - toe_right[1]) ** 2))

            distance_knee_toe_right = abs(knee_position_right - toe_position_right)
            if distance_knee_toe_right < 1.1 * foot_length_right:  # EXPERT ADVICE
                color_knee_right = color_green
            elif distance_knee_toe_right < 1.3 * foot_length_right:  # EXPERT ADVICE
                color_knee_right = color_yellow
            else:
                color_knee_right = color_red

            right_leg_plot = plot(point_knee_right, color_knee_right, abs(knee_position_right - toe_position_right), img)

            centroid_thigh_left = find_centroid(23, 25, lmlist)
            centroid_thigh_right = find_centroid(23, 26, lmlist)

            # # bounding box
            # toe_1_position = find_point_position(29, lmlist)
            # toe_2_position = find_point_position(31, lmlist)
            # toe_3_position = find_point_position(30, lmlist)
            # toe_4_position = find_point_position(32, lmlist)
            #
            # if toe_1_position > toe_2_position:  # left view
            #     rect_point_1 = int(toe_1_position[0] * 1.15), toe_1_position[1]
            #     rect_point_4 = int(toe_2_position[0] * 0.85), 10
            #     # distance_ear_and_bounding_box = (ear_position_left[0] - rect_point_1[0])
            #     # distance_hip_and_bounding_box = (left_hip[0] - rect_point_4[0])
            #     cv2.rectangle(img, rect_point_1, rect_point_4, color_green, 1, cv2.LINE_AA)
            # else:  # right view
            #     rect_point_1 = int(toe_3_position[0] * 0.85), toe_3_position[1]
            #     rect_point_4 = int(toe_4_position[0] * 1.15), 10
            #     # distance_ear_and_bounding_box = (ear_position_left[0] - rect_point_1[0])
            #     # distance_hip_and_bounding_box = (right_hip[0] - rect_point_4[0])
            #     cv2.rectangle(img, rect_point_1, rect_point_4, color_green, 2, cv2.LINE_AA)

            pose1 = results.pose_landmarks.landmark

    #imS = cv2.resize(img, (960, 540))
    writer.write(img)
    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
