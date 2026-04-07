import cv2 as cv
import numpy as np
from utils import *
from data import *
from find_targets import *
from debug_tools import *
import json

def find_tracks(path, draw = False):

    ref_unbonded = "../ModulePictures/Ref_img_unbonded.jpg"
    ref_bonded = "../ModulePictures/Ref_img_bonded.jpg"

    targets_dst = find_targets_wired(path)
    targets_ref = find_targets_wired(ref_bonded) # Changer par la ref

    H1 = cv.findHomography(targets_ref, targets_dst, cv.RANSAC)[0]
    H2 = compute_homography_center(cv.imread(ref_unbonded),cv.imread(ref_bonded))

    tracks = []
    if draw :
        img = cv.imread(path).copy()

    with open("REF_TRACKS.json") as f:
        data = json.load(f)
        for track_idx, track in data.items():

            track_bonded = warp_points(track, H2).astype(np.int32)
            track_dst = warp_points(track_bonded, H1).astype(np.int32)
            tracks.append(track_dst)
    
            if draw :
                if len(track_dst)==2 : 
                    cv.rectangle(img,track_dst[0], track_dst[1], (0, 255, 0), 3)
                elif len(track_dst)>0 :
                    cv.polylines(img,[np.array(track_dst).reshape((-1,1,2))], True, (0, 255, 0), 3)

    if draw :
        imS = cv.resize(img, (2000, 2000)) 
        cv.imshow("result",imS)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return tracks