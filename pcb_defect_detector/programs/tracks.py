import cv2 as cv
import numpy as np
from programs.utils import *
from programs.data import *
from programs.find_targets import *
from programs.debug_tools import *
import json

def find_tracks(path, draw = False, verbose_lv = 0, override_targets=False, read_targets=False):
    """
    Finds the location of the tracks for a given image, from their location on the reference image.

    Arguments:
    path - str: input path for image
    draw - bool: whether the function should show what it has computed.

    Returns:
    tracks - list of list of points: the positions of each track on the input image
    """
    ref_unbonded = "reference/Ref_img_unbonded.jpg"
    ref_bonded = "reference/Ref_img_bonded.jpg"

    with open("programs/REF_TARGETS.json") as f:
        data = json.load(f)
        targets_ref = np.array(data)

    if read_targets :
        with open("../ModuleData/targets_pos.json") as f:
            data = json.load(f)
            targets_dst = np.array(data)
    else :
        if not override_targets:
            if verbose_lv>1 : console.log(f"Recherche de la position des mires sur l'image...")
            targets_dst = find_targets_wired(path, verbose_lv=verbose_lv, img_name = "Image câblée")        
            if verbose_lv>1 : console.log(f"Positions des mires trouvées.")
        else:
            #Logique pour sélectionner chacune des mires à la main
            cabled_img = cv.imread(path)
            H, W = cabled_img.shape[:2]
            
            targets_dst = []

            sliceparams = [(50, 550, 750, 1350),  
                    (H-550, H-50, 750, 1350),
                    (2100, 3100, 900, 1700), 
                    (2100, 3100, 900, 1700), 
                    (50, 550, W-1350, W-750),
                    (H-550, H-50, W-1350, W-750), 
                    (2100, 3100, W-1700, W-900), 
                    (2100, 3100, W-1700, W-900)]
            
            for beg1, end1, beg2, end2 in sliceparams:
                target = magnifying_glass_targets(cabled_img[beg1:end1,beg2:end2])
                if not target:
                    raise Exception('Program Aborted.')
                targets_dst.append((target[0]+beg2, target[1]+beg1))
            
            targets_dst = np.array(targets_dst)

            if targets_dst[2,1] > targets_dst[3,1] :
                targets_dst[2,1], targets_dst[3,1] = targets_dst[3,1], targets_dst[2,1]
            if targets_dst[6,1] > targets_dst[7,1] :
                targets_dst[6,1], targets_dst[7,1] = targets_dst[7,1], targets_dst[6,1]

    if verbose_lv>1 : console.log(f"Calcul des homographies...")
    H1 = cv.findHomography(targets_ref, targets_dst, cv.RANSAC)[0]
    H2 = compute_homography_center(cv.imread(ref_unbonded),cv.imread(ref_bonded)) # TODO : ne pas le recalculer à chaque fois
    if verbose_lv>1 : console.log(f"Homographies calculées.")

    tracks = {}
    if draw :
        img = cv.imread(path).copy()

    with open("programs/REF_TRACKS.json") as f:
        data = json.load(f)
        for track_idx, track in data.items():
            
            if len(track) == 2 :
                x0, y0 = track[0]
                x1, y1 = track[1]
                track = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])

            track_bonded = warp_points(track, H2).astype(np.int32)
            track_dst = warp_points(track_bonded, H1).astype(np.int32)
            tracks[int(track_idx[5:])] = track_dst
    
            if draw :
                if len(track_dst)>0 :
                    cv.polylines(img,[np.array(track_dst).reshape((-1,1,2))], True, (0, 255, 0), 3)
    if verbose_lv>1 : console.log(f"Transposition des pistes effectuée.")

    if draw :
        imS = cv.resize(img, (2000, 2000)) 
        cv.imshow("result",imS)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return tracks