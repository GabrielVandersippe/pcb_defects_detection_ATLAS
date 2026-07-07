import cv2 as cv
import numpy as np
import json
from programs.find_targets import *
from programs.debug_tools import *

def find_pads (path, draw = False, verbose = 0, config = {}):

    lang = config["language"]
    if not verbose:
        verbose = config['verbose']

    with open("programs/REF_CORNERS.json") as f:
        data = json.load(f)
        corners_ref = np.array(data)

    img = cv.imread(path).copy()
    #img = cv.imread("reference/Ref_img_bonded.jpg").copy()

    shape = img.shape
    
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
    # print(corners)

    # Corners for P1004 (for test purposes)
    # corners = np.array([[818, 126], [825, shape[0] - 1000 + 814], [shape[1] - 1000 + 381, shape[0] - 1000 + 809], [shape[1] - 1000 + 379, 121]])

    if verbose>1 : console.log(f"Calcul des homographies...")
    H = cv.findHomography(corners_ref, corners, cv.RANSAC)[0]
    if verbose>1 : console.log(f"Homographies calculées.")

    with open("programs/REF_PADS_GROUPED.json") as f:
        pads_ref = json.load(f)

    nb_pads = 198

    pads = {}

    with open("programs/REF_PADS.json") as f:
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
    
    return pads