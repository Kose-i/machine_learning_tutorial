import cv2
import numpy as np
import IPython

def get_frame(cap, scaling_factor):
    r, frame = cap.read()
    if not r: return None
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame
def show_image(image):
    _, png = cv2.imencode('.png', image)
    i = IPython.display.Image(data=png)
    IPython.display.clear_output(wait=True)
    IPython.display.display(i)

cap = cv2.VideoCapture('datasets/ed.mp4')
scaling_factor = 1

lower = np.array([ 0,  30,  60])
upper = np.array([50, 255, 255])

try:
    while True:
        frame = get_frame(cap, scaling_factor)
        if frame is None: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)
        img2 = cv2.hconcat([frame, img_median_blurred])
        show_image(img2)
except KeyboardInterrupt:
    print('Interrupted')

cap.release()

cap = cv2.VideoCapture('datasets/bbb.mp4')
scaling_factor = 1

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

history = 100
learning_rate = 1.0 / history

try:
    while True:
        frame = get_frame(cap, scaling_factor)
        if frame is None: break
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img2 = cv2.hconcat([frame, frame & mask])
        show_image(img2)
except KeyboardInterrupt:
    print('Interrupted')

cap.release()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RectInput:
    def __init__(self, image):
        self.fig, self.ax = plt.subplots()
        self.pev = None
        self.rect = None
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self.on_release)
        self.bbox = None
        plt.show()
    def on_press(self, ev):
        self.pev = ev
        if self.rect is None:
            self.rect = patches.Rectangle((ev.xdata, ev.ydata), 1, 1, color='white', fill=False)
            self.ax.add_patch(self.rect)
        else:
            self.rect.set_bounds(ev.xdata, ev.ydata, 1, 1)
    def on_release(self, ev):
        self.pev = None
    def on_move(self, ev):
        if self.pev is None: return
        self.bbox = (int(min(self.pev.xdata,  ev.xdata)),
                     int(min(self.pev.ydata,  ev.ydata)),
                     int(abs(self.pev.xdata - ev.xdata)),
                     int(abs(self.pev.ydata - ev.ydata)))
        self.ax.set_xlabel('bbox {} {} {} {}'.format(*self.bbox))
        self.rect.set_bounds(*self.bbox)
        self.fig.canvas.draw()

cap = cv2.VideoCapture('datasets/bbb.mp4')
scaling_factor = 1.0
first_frame = get_frame(cap, scaling_factor)
cap.release()

rinput = RectInput(first_frame)


x,y,w,h = rinput.bbox
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
hsv_roi = hsv[y:y+h, x:x+w]
hist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

track_window = (x, y, w, h)
cap = cv2.VideoCapture('datasets/bbb.mp4')
try:
    while True:
        frame = get_frame(cap, scaling_factor)
        if frame is None: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_backproj = cv2.calcBackProject([hsv], [0], hist, [0,180], 1)
        track_box, track_window = cv2.CamShift(hsv_backproj, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        bp = cv2.cvtColor(hsv_backproj, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(frame, track_box, (0, 255.0), 2)
        img2 = cv2.hconcat([bp, frame])
        show_image(img2)
except KeyboardInterrupt:
    print('Interrupted')

cap.release()

NUM_FRAMES_TO_TRACK = 5
NUM_FRAMES_JUMP = 2

cap = cv2.VideoCapture('datasets/ed.mp4')
scaling_factor = 1
TRACKING_PARAMS = {"winSize": (11,11), "maxLevel": 2, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

tracking_paths = []
frame_index = 0
prev_gray = None

try:
    while True:
        frame = get_frame(cap, scaling_factor)
        if frame is None: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = frame_gray
            continue
        if frame_index % NUM_FRAMES_JUMP == 0:
            mask = np.ones_like(prev_gray)
            for x,y in [tp[-1] for tp in tracking_paths]:
                cv2.circle(mask, (x,y), 6, 0, -1)
            feature_points = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
            if feature_points is not None:
                for x,y in feature_points.reshape(-1, 2):
                    tracking_paths.apend([(x,y)])
        if len(tracking_paths) > 0:
            feature_points_0 = np.float32([tp[-1] for tp in tracking_paths]).reshape(-1,1,2)
            feature_points_1, _, _     = cv2.calcOpticalFlowPyrLK(prev_gray,  frame_gray, feature_points_0, None, **TRACKING_PARAMS)
            feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray,  feature_points_1, None, **TRACKING_PARAMS)
            diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1, 2).max(axis=1)
            good_points = diff_feature_points < 1
            new_tracking_paths = []
            for tp, (x,y), good_points_flag in zip(tracking_paths, feature_points_1.reshape(-1,2), good_points):
                if not good_points_flag: continue
                tp.append(x,y)
                if len(tp) > NUM_FRAMES_TO_TRACK:
                    del tp[0]
                new_tracking_paths.append(tp)
                cv2.circle(frame, (x,y), 3, (0, 255, 0), -1)
                cv2.polylines(frame, [np.int32(tp)], False, (0, 150, 0))
            tracking_paths = new_tracking_paths
        show_image(frame)
        frame_index += 1
        prev_gray = frame_gray
except KeyboardInterrupt:
    print('Interrupted')

cap.release()
