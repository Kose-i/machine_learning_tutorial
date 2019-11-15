import cv2
import IPython.display

def show_image(image):
    _, png = cv2.imencode('.png', image)
    i = IPython.display.Image(data=png)
    IPython.display.clear_output(wait=True)
    IPython.display.display(i)

def get_frame(cap, scaling_factor):
    r, frame = cap.read()
    if not r: return None
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

cap = cv2.VideoCapture('datasets/bbb.mp4')

import time
try:
    while True:
        frame = get_frame(cap, 1.0)
        if frame is None: break
        show_image(frame)
        time.sleep(0.03)
    print('Finished')
except KeyboardInterrupt:
    print('Interrupted')

cap.release()

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames_2 = cv2.absdiff(cur_frame,  next_frame)
    return cv2.bitwise_and(diff_frames_1, diff_frames_2)

def get_gray_frame(cap, scaling_factor):
    frame = get_frame(cap, scaling_factor)
    if frame is None: return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

cap = cv2.VideoCapture('datasets/bbb.mp4')
scaling_factor = 1

prev_frame = get_gray_frame(cap, scaling_factor)
cur_frame  = get_gray_frame(cap, scaling_factor)
next_frame = get_gray_frame(cap, scaling_factor)
try:
    while True:
        diff = frame_diff(prev_frame, cur_frame, next_frame)
        show_image(diff)
        prev_frame = cur_frame
        cur_frame  = next_frame
        next_frame = get_gray_frame(cap, scaling_factor)
        if next_frame is None: break
except KeyboardInterrupt:
    print('Interrupted')

cap.release()
