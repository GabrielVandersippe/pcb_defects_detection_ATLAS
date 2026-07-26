import cv2 as cv
import numpy as np
import json
from programs.find_targets import *
from programs.debug_tools import *

def find_pads (path, draw = False, verbose = 0, config = {}, read_corners=False, full_image_mode=False):

    lang = config["language"]
    if not verbose:
        verbose = config['verbose']

    with open("reference/REF_CORNERS.json") as f:
        data = json.load(f)
        corners_ref = np.array(data)

    img = cv.imread(path).copy()

    shape = img.shape

    if path[-1] == "/" :
        file_name = (path.split("/")[-2]).split(".")[0]
    else :
        file_name = (path.split("/")[-1]).split(".")[0]
    
    if read_corners :
        corners = np.loadtxt(f"../output/data/corners_pos_{file_name}.txt", dtype=int, delimiter=",")

    elif not full_image_mode:
        message_mg_pads()
        corner_GA1 = magnifying_glass_pads(img[:1000,:1000])
        if not corner_GA1:
            raise Exception('Program Aborted.')
        corner_GA2 = magnifying_glass_pads(img[-1000:,:1000])
        if not corner_GA2:
            raise Exception('Program Aborted.')
        corner_GA3 = magnifying_glass_pads(img[-1000:,-1000:])
        if not corner_GA3:
            raise Exception('Program Aborted.')
        corner_GA4 = magnifying_glass_pads(img[:1000,-1000:])
        if not corner_GA4:
            raise Exception('Program Aborted.')
        
        corners = np.array([corner_GA1, corner_GA2, corner_GA3, corner_GA4])
        corners[1,1] += shape[0] - 1000
        corners[2,0] += shape[1] - 1000
        corners[2,1] += shape[0] - 1000
        corners[3,0] += shape[1] - 1000
        
    else: #full_image_mode active
        message_mg_pads()
        corner_GA1 = magnifying_glass_pads(img)
        if not corner_GA1:
            raise Exception('Program Aborted.')
        corner_GA2 = magnifying_glass_pads(img)
        if not corner_GA2:
            raise Exception('Program Aborted.')
        corner_GA3 = magnifying_glass_pads(img)
        if not corner_GA3:
            raise Exception('Program Aborted.')
        corner_GA4 = magnifying_glass_pads(img)
        if not corner_GA4:
            raise Exception('Program Aborted.')
        
        corners = np.array([corner_GA1, corner_GA2, corner_GA3, corner_GA4])

    np.savetxt(f"../output/data/corners_pos_{file_name}.txt", corners, fmt='%d', delimiter=',')

    if verbose>1 : console.log(f"Calcul des homographies...")
    H = cv.findHomography(corners_ref, corners, 0)[0]
    if verbose>1 : console.log(f"Homographies calculées.")

    pads = {}

    with open("reference/REF_PADS.json") as f:
        data = json.load(f)
        for pad_idx, pad in data.items():
            
            if len(pad) == 2 :
                x0, y0 = pad[0]
                x1, y1 = pad[1]
                pad = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])

            pad_bonded = warp_points(pad, H).astype(np.int32)
            pads[int(pad_idx[3:])] = pad_bonded
    
            if draw :
                if len(pad_bonded)>0 :
                    print(pad_bonded)
                    cv.polylines(img,[np.array(pad_bonded).reshape((-1,1,2))], True, (0, 255, 0), 3)
    if verbose>1 : console.log(f"Transposition des pads effectuée.")

    if draw :
        magnifying_glass(img)
    
    return pads