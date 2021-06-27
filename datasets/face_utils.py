"""
http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/
"""

from PIL import Image
import numpy as np
import torch
import imutils
import cv2


def align_face(frame, face_coords, landmarks):
    landmarks = np.array(landmarks)
    
    roi = { 
            'nose': slice(27, 31),
            'nose_point': slice(30, 31),
            'nostril': slice(31, 36),
            'eye1': slice(36, 42),
            'eye2': slice(42, 48)
        }

    def get_roi_mid_point(roi):
        x, y, w, h = cv2.boundingRect(landmarks[roi])
        mid_x = x + w // 2
        mid_y = y + h // 2
        return mid_x, mid_y

    left_eye = get_roi_mid_point(roi['eye1'])
    right_eye = get_roi_mid_point(roi['eye2'])

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y

    try:
        angle = np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
    except ZeroDivisionError:
        angle = 0
        
    x1, y1, x2, y2 = face_coords
    img = frame[:, y1: y2, x1: x2].permute(1, 2, 0).numpy()

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated
    
    # nose = landmarks[roi['nose_point']]
    # if left_eye_y > right_eye_y:
    #     A = (right_eye_x, left_eye_y)
    #     # Integer -1 indicates that the image will rotate in the clockwise direction
    #     direction = -1 
    # else:
    #     A = (left_eye_x, right_eye_y)
    #     # Integer 1 indicates that image will rotate in the counter clockwise  
    #     # direction
    #     direction = 1 

    # # cv2_imshow(rotated)
    # # cv2.imshow('rotated', rotated)
    # # center_pred = x1 + ((x2 - x1) // 2), y1 + ((y2 - y1) // 2)
    # # length_line1 = distance(center_of_forehead, nose)
    # # length_line2 = distance(center_pred, nose)
    # # length_line3 = distance(center_pred, center_of_forehead)
    # # cos_a = cosine_formula(length_line1, length_line2, length_line3)
    # # angle = np.arccos(cos_a)
    # # rotated_point = rotate_point(nose, center_of_forehead, angle)
    # # rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
    # # if is_between(nose, center_of_forehead, center_pred, rotated_point):
    # #     angle = np.degrees(-angle)
    # # else:
    # #     angle = np.degrees(angle)

    # # face = frame[:, y1: y2, x1: x2]        
    # # img = frame.permute(1, 2, 0).numpy().astype(dtype=np.uint8)
    # # img = Image.fromarray(img)
    # # img = np.array(img.rotate(-angle))
    # # cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    # for x, y in landmarks[roi['eye1']]:
    #     cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # print()
    # exit(0)


def align_and_crop_face(frame, face_coords, landmarks):
    face = align_face(frame, face_coords, landmarks)    
    
    return torch.from_numpy(face).permute(2, 0, 1)
