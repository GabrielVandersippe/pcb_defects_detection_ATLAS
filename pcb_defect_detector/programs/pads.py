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

    cimg = img.copy()

    for GA, ref in pads_ref.items():
        tl, tr, br, bl = warp_points(ref, H) #top left, top right, bottom right, bottom left
        delta_height = (bl[0] - tl[0])/nb_pads
        delta_width = (bl[1] - tl[1])/nb_pads

        print(delta_height, delta_width)

        pads[GA] = {}

        for i in range(nb_pads):
            key = GA.strip('GA')
            nb_str = str(i)
            key = key + (3-len(nb_str))*'0' + nb_str
            pads[GA][key] = [(int(tl[0]+i*delta_height), int(tl[1]+i*delta_width)), 
                                (int(tr[0]+i*delta_height), int(tr[1]+i*delta_width)),
                                (int(tr[0]+(i+1)*delta_height), int(tr[1]+(i+1)*delta_width)),
                                (int(tl[0]+(i+1)*delta_height), int(tl[1]+(i+1)*delta_width))]

            cv.rectangle(cimg, pads[GA][key][0], pads[GA][key][2], [0,0,255],1)

    magnifying_glass(cimg)

    if verbose>1 : console.log(f"Transposition des pads effectuée.")
    
    return pads