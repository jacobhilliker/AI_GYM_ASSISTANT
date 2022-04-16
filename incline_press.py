import cv2
import mediapipe as mp
import time

from util import *

# called by main function
def check_incline_press():

    global reps, good_reps, start_rep, mid_rep, end_rep, past_mid_rep, is_user_ready

    # initialize rep tracking
    reps = 0
    good_reps = 0
    
    # initialize weight marker tracking
    tracker = cv2.TrackerCSRT_create()
    marker_x, marker_y, marker_w, marker_h = 0, 0, 0, 0

    # initialize colors for arm status
    upper_arm_color = COLOR_BLACK
    arm_color = COLOR_BLACK
    start_color = COLOR_BLACK
    
    # tracks good, OK, and poor frames at beginning, middle, and end of rep
    start_rep = [0, 0, 0]
    mid_rep = [0, 0, 0]
    end_rep = [0, 0, 0]

    past_mid_rep = False

    # avoids counting "bad" frames before first rep starts
    is_user_ready = False

    # instantiate pose module
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # start video capture
    cap = cv2.VideoCapture(0)

    # time performance tracking
    last_time, current_time = 0, 0

    while cap.isOpened():

        ok, img = cap.read()
        
        '''
        # aruco tracking
        found_markers = find_aruco_markers(img)

        if len(found_markers[0]) != 0:
            marker_x, marker_y, marker_w, marker_h = plot_aruco_markers(found_markers, img)
            try:
                ok = tracker.init(img, (marker_x, marker_y, marker_w, marker_h))
            except:
                pass
        else:
            try:
                ok, (marker_x, marker_y, marker_w, marker_h) = tracker.update(img)
            except:
                pass

        marker_centroid = find_centroid(marker_x, marker_y, marker_h, marker_w)
        cv2.circle(img, marker_centroid, 4, COLOR_GREEN, 3)
        '''        

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

                # bottom of the rep
                if wrist_point[1] >= ear_point[1]:

                    # check and plot upper arm angle (expert advice)
                    elbow_shoulder_dx = abs(elbow_point[0] - shoulder_point[0])
                    torso_dx = abs(shoulder_point[0] - hip_point[0])

                    upper_arm_color = classify_upper_arm(shoulder_point, elbow_point, hip_point, elbow_shoulder_dx, torso_dx)

                    if incline_color != COLOR_RED and upper_arm_color != COLOR_RED:
                        is_user_ready = True

                    upper_arm_line = get_line_segment(shoulder, elbow, landmarks)
                    plot_line(upper_arm_line[0], upper_arm_line[1], upper_arm_color, img)

                    # update indicators for start position and arm extension
                    if past_mid_rep:
                        if is_good_mid():
                            arm_color = COLOR_GREEN
                        else:
                            arm_color = COLOR_YELLOW
                    else:
                        arm_color = COLOR_BLACK
                        start_color = upper_arm_color


                # top of the rep
                else:

                    # check and plot full arm angle (expert advice)
                    arm_extension_points = (shoulder_point, elbow_point, wrist_point)
                    arm_extension_angle = calculate_angle(arm_extension_points)

                    arm_elevation_points = (hip_point, shoulder_point, wrist_point)
                    arm_elevation_angle = calculate_angle(arm_elevation_points)

                    # might not work with actual weights; might need to use arUco marker instead of wrist
                    arm_color = classify_arm(arm_extension_angle, arm_elevation_angle)
                    plot_line(shoulder_point, elbow_point, arm_color, img)
                    plot_line(elbow_point, wrist_point, arm_color, img)

                    # update indicator for upper arm angle
                    if is_good_start():
                        start_color = COLOR_GREEN
                    else:
                        start_color = COLOR_YELLOW

            # ---------- REP TRACKING ----------

            # set global variable for mid rep if mid rep has been maintained for select number of frames
            past_mid_rep = sum_over(mid_rep) >= 20

            # count rep as complete
            if past_mid_rep and sum_over(end_rep) >= 20:
                    
                reps += 1
                    
                if is_good_rep():
                    good_reps += 1

                print(f'\nrep: {reps}')
                print(start_rep)
                print(mid_rep)
                print(end_rep)
                    
                reset_rep()
                upper_arm_color = COLOR_BLACK
                arm_color = COLOR_BLACK

            # label image with numbers of reps and good reps
            if is_user_ready:
            
                plot_label((12, 36), 'Reps: ', COLOR_BLACK, img, scale=1.5)
                plot_label((210, 38), reps, COLOR_BLACK, img, scale=1.5)
                plot_label((12, 72), 'Good Reps: ', COLOR_BLACK, img, scale=1.5)
                plot_label((210, 74), good_reps, COLOR_BLACK, img, scale=1.5)

                plot_label((12,108), 'Start Position', COLOR_BLACK, img, scale=1.5)
                plot_rectangle(img, '', (180, 180), (210, 90), (230, 110), start_color)
                plot_label((12,144), 'Arm Extension', COLOR_BLACK, img, scale=1.5)
                plot_rectangle(img, '', (180, 180), (210, 126), (230, 146), arm_color)

        # track FPS
        current_time = time.time()
        if last_time != 0:
            plot_label((12, 180), f'FPS: {int(1 / (current_time - last_time))}', COLOR_BLACK, img, scale=1.5)
        else:
            plot_label((12, 180), 'FPS: 0', COLOR_BLACK, img, scale=1.5)
        last_time = current_time

        # show final image
        cv2.imshow("AI Gym Assistant", img)

        # quit with 'q' key
        if cv2.waitKey(5) == ord('q'):
            print(f'\nreps: {reps}\n')
            break

    cap.release()
    cv2.destroyAllWindows()

'''
Parameters:
    arm_extension_angle: the angle formed by the user's most visible shoulder, elbow, and wrist.
    arm_elevation_angle: the angle formed by the user's most visible hip, shoulder, and wrist.
Determines if the user's arm is correctly extended using a scoring system.
Returns a color reflecting the user's score.
'''
def classify_arm(arm_extension_angle, arm_elevation_angle):

    score = 0

    if arm_extension_angle >= 145 and arm_extension_angle <= 170:
        score += 2
    elif arm_extension_angle >= 130 and arm_extension_angle <= 185:
        score += 1

    if arm_elevation_angle >= 90 and arm_elevation_angle <= 110:
        score += 2
    elif arm_elevation_angle >= 80 and arm_elevation_angle <= 120:
        score += 1

    # assess scoring to decide color
    arm_color = COLOR_RED

    if score == 4:
        arm_color = COLOR_GREEN
    elif score >= 2:
        arm_color = COLOR_YELLOW

    if is_user_ready:
        update_rep(arm_color, mid_rep)

    return arm_color

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
Determines if arm angle at the bottom of the rep is good, OK, or poor.
Returns a color to be plotted on the image.
'''
def classify_upper_arm(shoulder_point, elbow_point, hip_point, elbow_shoulder_dx, torso_dx):

    global past_mid_rep

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

    if is_user_ready:

        if past_mid_rep:
            update_rep(upper_arm_color, end_rep)
        else:
            update_rep(upper_arm_color, start_rep)

    return upper_arm_color


def is_good_start():
    return start_rep[GOOD] > start_rep[OK] and start_rep[GOOD] > start_rep[POOR]


def is_good_mid():
    return mid_rep[GOOD] >= 3 and mid_rep[GOOD] > mid_rep[POOR]


def is_good_end():
    return end_rep[GOOD] > end_rep[OK] and end_rep[GOOD] > end_rep[POOR]


def is_good_rep():

    good_rep_parts = 0

    if is_good_start():
        good_rep_parts += 1
                    
    if is_good_mid():
        good_rep_parts += 1

    if is_good_end():
        good_rep_parts += 1

    return good_rep_parts > 1


def update_rep(color, rep_stage):

    if color == COLOR_GREEN:
        rep_stage[GOOD] += 1
    elif color == COLOR_YELLOW:
        rep_stage[OK] += 1
    elif color == COLOR_RED:
        rep_stage[POOR] += 1


def reset_rep():
    
    global start_rep, mid_rep, end_rep

    start_rep = [0, 0, 0]
    mid_rep = [0, 0, 0]
    end_rep = [0, 0, 0]


# main function
if __name__ == '__main__':
    check_incline_press()