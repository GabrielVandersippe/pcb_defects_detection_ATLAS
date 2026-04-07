from Programs.output import print_success, print_error, print_info
from Programs.count import extract_serial_number, iref_trim
from Programs.map import bounding_map_pads_pistes, bounding_map_without_trim, bounding_map_trim
from Programs.wire_detection import *
import json

bounding_map_without_trim()

def run_check (path, draw = False) :


    ok = True


    img = cv.imread(path)
    ROI = find_ROI(img)
    lmask = img[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]]
    rmask = img[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]]

    y_left, x_left, y_right, x_right = wire_pos(img)

    y_left_ROI = y_left - ROI[1][1]
    x_left_ROI = x_left - ROI[0][0]

    y_right_ROI = y_right - ROI[3][1]
    x_right_ROI = x_right - ROI[2][0]


    short_list_left, endpoints_left, labels_left = find_shorts(lmask, 'left', y_left_ROI, x_left_ROI, draw)
    short_list_right, endpoints_right, labels_right = find_shorts(rmask, 'right', y_right_ROI, x_right_ROI, draw)

    cimg = img.copy()


    ## ---- CHECKS POUR VERIFIER LE CABLAGE ----

    # 1. Vérifier si tous les fils sont présents : 
    with open("ModulePictures/iref_trim_per_module_v2.json", "r") as f:
        data = json.load(f)
        n_expected = expected_wire_number(extract_serial_number(path),data)
        n_detected = len(y_left) + len(y_right)
        print("Wires expected : " + str(n_expected))
        print("Wires detected : " + str(n_detected))
        ok = n_expected == n_detected

    # Ajouter Logique pour ne faire la suite que si on a le bon nombre de câbles !

    serial_number = extract_serial_number(path)
    trim_nb = iref_trim(serial_number, data)
    bounding_map_trim(trim_nb)
    wires = bounding_map_pads_pistes()

    nb_shorts = 0
    tracks = find_tracks(path) # TODO : ajouter les tracks

    # 2. Vérifier si des fils qui ne se touchent pas sont bien câblés.
    for label,endpoint in endpoints_left.items():
        if endpoint == None:
            nb_shorts += 1
            # Mettre le fil en rouge, TODO ou en orange si OK en termes de pistes d'arivées.
            cimg[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = [0,0,255]
            
        else :
            (endpoint_location, wire_idx) = endpoint
            cv.circle(cimg, (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), 4, (255, 0, 0), 2)
            
            # 3. TODO Vérifier si les fils vont bien au bon endroit
            # is_in_right_track = cv.pointPolygonTest(tracks[wires[wire_idx]], endpoint_location)


    for label, endpoint in endpoints_right.items():
        if endpoint == None:
            nb_shorts += 1
            # Mettre le fil en rouge, TODO ou en orange si OK en termes de pistes d'arivée.
            cimg[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = [0,0,255]

        else :
            (endpoint_location, wire_idx) = endpoint
            cv.circle(cimg, (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), 4, (255, 0, 0), 2) 
            
            # 3. TODO Vérifier si les fils vont bien au bon endroit

    print(f'{nb_shorts} courts-circuits potentiels détectés.')
    for short in short_list_left: 
        cv.circle(cimg, (short[0] + ROI[0][0], short[1] + ROI[1][1]), 4, (0, 255, 0), 2)
    for short in short_list_right: 
        cv.circle(cimg, (short[0] + ROI[2][0], short[1] + ROI[3][1]), 4, (0, 255, 0), 2)

    magnifying_glass(cimg)
    


    if ok :
        print_success("Module correctement cablé")
    else :
        print_error("Module incorrect")
        print_info("Nombre des fils : " + "/" + str(n_expected)) # insérer le nombre de fils comptés
        # afficher les zones (1/2/3/4) des fils manquants s'il y en a
        # afficher les zones des fils mal branchés s'il y en a
        # afficher les zones des fils qui se touchent s'il y en a
        # afficher des images du cablage