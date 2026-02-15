import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from data import *
from find_targets import *


def draw_area(img, pt1, pt2):
    cv.rectangle(img, pt1, pt2, (0, 255, 0), 20)
    plt.imshow(img)
    plt.show()




# -----------------------------------------------
# Magnifying glass for an image using OpenCV : 
# -----------------------------------------------
drawing = False # true if mouse is pressed
ix,iy = -1,-1
patch_size=100
zoom_scale=4

def mouse_callback(event, x, y, flags, param):
    global ix,iy,drawing, patch_size, zoom_scale
    
    state = param
    img = state['current_view']

    # Define the ROI 
    x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
    x2, y2 = min(img.shape[1], x + patch_size // 2), min(img.shape[0], y + patch_size // 2)
    
    roi = img[y1:y2, x1:x2]
    
    if roi.size > 0:
        zoom_view = cv.resize(roi, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv.INTER_NEAREST)
        h_z, w_z = zoom_view.shape[:2]
        
        #Write the coordinates on the zoomed view
        coord_text = f"X:{x} Y:{y}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        (text_w, text_h), _ = cv.getTextSize(coord_text, font, font_scale, thickness)

        # Calculate position (Bottom Right with 10px padding)
        text_x = w_z - text_w - 10
        text_y = h_z - 10

        cv.rectangle(zoom_view, (text_x - 5, text_y - text_h - 5), (w_z, h_z), (0, 0, 0), -1)
        cv.putText(zoom_view, coord_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)

        # Draw a crosshair in the center of the zoom window
        cv.line(zoom_view, (w_z//2, 0), (w_z//2, h_z), (0, 255, 0), 1)
        cv.line(zoom_view, (0, h_z//2), (w_z, h_z//2), (0, 255, 0), 1)
        
        cv.imshow("Magnifying Glass", zoom_view)
    

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x_start, x_end = sorted([ix, x])
        y_start, y_end = sorted([iy, y])

        # Check if selected area large enough
        if (x_end - x_start) > 15 and (y_end - y_start) > 15:
            state['offset_x'] += x_start
            state['offset_y'] += y_start

            state['current_view'] = img[y_start:y_end, x_start:x_end]


def magnifying_glass(path):
    img = cv.imread(path)

    state = {
        'current_view': img.copy(),
        'offset_x': 0,
        'offset_y': 0
    }

    cv.namedWindow("Main PCB View", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Main PCB View", mouse_callback, param=state)

    print("--- Magnifying Tool Instructions ---")
    print("Move mouse to zoom | Left-Click to save point | Press 'q' to finish")

    while True:
        cv.imshow("Main PCB View", state['current_view'])
        key = cv.waitKey(1) & 0xFF 
        
        if key == ord('q'):
            break
        elif key == ord('r'): # Reset functionality
            state['current_view'] = img.copy()
            state['offset_x'], state['offset_y'] = 0, 0
        

    cv.destroyAllWindows()