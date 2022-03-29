# importing modules
import cv2
import numpy as np
import cv2.aruco as aruco
import os

def calculate_angle(a):
    radians = np.arctan2(a[2][1] - a[1][1], a[2][0] - a[1][0]) - np.arctan2(a[0][1] - a[1][1], a[0][0] - a[1][0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle)

def find_line_length(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)


def find_midpoint(id1, id2,landmarks):
    point1 = (landmarks[id1][1], landmarks[id1][2])
    point2 = (landmarks[id2][1], landmarks[id2][2])
    midpoint_x = int((landmarks[id1][1] + landmarks[id2][1]) / 2)
    midpoint_y = int((landmarks[id1][2] + landmarks[id2][2]) / 2)
    midpoint = (midpoint_x, midpoint_y)
    return midpoint


'''
def extend_line(id1, id2, landmarks):
    p1, p2 = get_line_segment(id1, id2, landmarks)
    x1, y1, x2, y2 = (p1[1], p1[2], p2[1], p2[2])
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    p1_ext = (x1 - delta)
'''

def find_point_position(id,landmarks):
    point = (landmarks[id][1], landmarks[id][2])
    return point


def find_positions(id1, id2, id3,landmarks):
    point1 = (landmarks[id1][1], landmarks[id1][2])
    point2 = (landmarks[id2][1], landmarks[id2][2])
    point3 = (landmarks[id3][1], landmarks[id3][2])
    return point1, point2, point3


def get_line_segment(id1, id2, landmarks):
    return (find_point_position(id1, landmarks), find_point_position(id2, landmarks))


def plot_point(point,color,img):
    cv2.circle(img, point, 5, color, cv2.FILLED)
    return None


def plot_lines_3points(pt1, pt2, pt3,img):
    points = np.array([(pt1), pt2, (pt3)])
    cv2.drawContours(img, [points], 0, (255, 255, 255), 2)


def plot(point, color,angle,img):
    cv2.line(img, point[0], point[1], color, 2)
    cv2.line(img, point[1], point[2], color, 2)

    cv2.circle(img, point[0], 2, color, cv2.FILLED)
    cv2.circle(img, point[1], 2, color, cv2.FILLED)
    cv2.circle(img, point[2], 2, color, cv2.FILLED)

    # if angle:
    cv2.putText(img, str(angle), point[1],
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return None


def plot_line(pt1, pt2, color, img):
    cv2.line(img, pt1, pt2, color, 2)


def plot_label(pt, label, color, img):
    label = str(label)
    cv2.putText(img, label, pt, cv2.FONT_HERSHEY_PLAIN, 2, color, 2, cv2.LINE_AA)


def plot_bar(angle, angle_limits,img):
    per = np.interp(angle, angle_limits, (0, 100))
    bar = np.interp(angle, angle_limits, (400, 120))
    # counter logic
    if per == 100:
        barcolor = (0, 255, 0)
    else:
        barcolor = (0, 0, 255)

    # Setup status box
    cv2.rectangle(img, (25, 120), (55, 400), barcolor, 2)
    cv2.rectangle(img, (25, int(bar)), (55, 400), barcolor, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (25, 110), cv2.FONT_HERSHEY_PLAIN, 2, barcolor, 1, cv2.LINE_AA)
    return per, bar


def plot_bar_horizontal(distance,img,thigh_half_length,color_Head_thigh):
    dis_mod = abs(distance)

    if distance <= 0:
        dis_mod = abs(distance)
        per = np.interp(dis_mod, (0, 2 * thigh_half_length), (-100, 0))
        bar = np.interp(dis_mod, (0, 2 * thigh_half_length), (330, 260))

        cv2.rectangle(img, (260, 20), (400, 40), color_Head_thigh, 2)
        cv2.rectangle(img, (330, 20), (int(bar), 40), color_Head_thigh, cv2.FILLED)
    elif distance > 0:
        per = np.interp(dis_mod, (2 * thigh_half_length, 0), (0, 100))
        bar = np.interp(dis_mod, (0, 2 * thigh_half_length), (330, 400))

        cv2.rectangle(img, (260, 20), (400, 40), color_Head_thigh, 2)
        cv2.rectangle(img, (330, 20), (int(bar), 40), color_Head_thigh, cv2.FILLED)


def find_aruco_markers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    # print(bboxs)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

'''
Returns x, y, width, and height of the bounding box of the LAST found arUco marker
'''
def plot_aruco_markers(arucofound,img):

    for bbox, id in zip(arucofound[0], arucofound[1]):
        top_left = bbox[0][0][0], bbox[0][0][1]
        top_right = bbox[0][1][0], bbox[0][1][1]
        bottom_right = bbox[0][2][0], bbox[0][2][1]
        bottom_left = bbox[0][3][0], bbox[0][3][1]

        marker_centroid = int((top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4), int((top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4)
        cv2.circle(img, (marker_centroid[0], marker_centroid[1]), 3, (255, 0, 0), 3)
        lx = int(top_left[0])
        ly = int(top_left[1])
        rx = int(bottom_right[0])
        ry = int(bottom_right[1])

        if rx == lx or ry == ly:
            bounding_box = (lx, ly, 100, 100)

        else:
            bounding_box = (lx, ly, (rx - lx), (ry - ly))

    return bounding_box

def graph_plot():
    os.system("plotting.py")