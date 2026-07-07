import numpy as np
import cv2 as cv
from programs.debug_tools import magnifying_glass_final_result, magnifying_glass_ref

dico = {}

def left_side () :
    # Tuned for left side of Ref_img_bonded.jpg

    first_pad = [[751, 199], [765, 209]]
    second_pad = [[752, 212], [765, 221]]
    last_pad = [[762, 5295], [775, 5304]]

    slope = (first_pad[0][0] - last_pad[0][0]) / (first_pad[0][1] - last_pad[0][1])
    dy = second_pad[0][1] - first_pad[0][1]
    lx = first_pad[1][0] - first_pad[0][0]
    ly = first_pad[1][1] - first_pad[0][1]

    gap = 64
    for i in range (2*198) :
        y = first_pad[0][1] + i*dy
        if i >= 198 :
            y += gap - dy
        y -= (i//5)*1.15
        x = round(first_pad[0][0] + slope * (y - first_pad[0][1]))
        y = round(y)
        if i >= 198 :
            i -= 198
            i += 2000
        else :
            i += 1000
        i += 1
        dico["PAD"+str(i)] = [[x, y], [x+lx, y+ly]]

def right_side () :
    # Tuned for right side of Ref_img_bonded.jpg

    first_pad = [[6125, 5285], [6139, 5294]]
    second_pad = [[6125, 5272], [6139, 5281]]
    last_pad = [[6116, 186], [6129, 195]]

    slope = (first_pad[0][0] - last_pad[0][0]) / (first_pad[0][1] - last_pad[0][1])
    dy = second_pad[0][1] - first_pad[0][1]
    lx = first_pad[1][0] - first_pad[0][0]
    ly = first_pad[1][1] - first_pad[0][1]

    gap = 63
    for i in range (2*198) :
        y = first_pad[0][1] + i*dy
        if i >= 198 :
            y -= gap + dy
        y += (i//5)*1.1
        x = round(first_pad[0][0] + slope * (y - first_pad[0][1]))
        y = round(y)
        if i >= 198 :
            i -= 198
            i += 4000
        else :
            i += 3000
        i += 1
        dico["PAD"+str(i)] = [[x, y], [x+lx, y+ly]]
    
right_side()

for key, values in dico.items() :
    print(str(chr(34)) + key + str(chr(34)) + ": " + str(values) + ",")

path = "reference/Ref_img_bonded.jpg"
img = cv.imread(path)
pads_mask = np.zeros(img.shape[:2], dtype=np.uint8)

for pad in dico.values():
    if len(pad) == 2 :
        x0, y0 = pad[0]
        x1, y1 = pad[1]
        pad = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])
    if len(pad)>0 :
        cv.polylines(pads_mask,[np.array(pad).reshape((-1,1,2))], True, 1, 1)

np.save("../temp/pads_mask.npy", pads_mask)

crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
non_crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
non_crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
np.save("../temp/crit_shorts_mask.npy", crit_shorts_mask)
np.save("../temp/non_crit_shorts_mask.npy", non_crit_shorts_mask)
np.save("../temp/crit_endpoints_mask.npy", crit_endpoints_mask)
np.save("../temp/non_crit_endpoints_mask.npy", non_crit_endpoints_mask)

#test = magnifying_glass_ref(path)
#print(test)
#magnifying_glass_final_result(path)
