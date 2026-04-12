from Programs.output import *
from Programs.count import extract_serial_number, iref_trim
from Programs.map import bounding_map_pads_pistes, bounding_map_without_trim, bounding_map_trim
from Programs.wire_detection import *
import time
import json

bounding_map_without_trim()

def run_check (path, draw = False) :

    ok = False

    img = cv.imread(path)
    ROI = find_ROI(img)
    lmask = img[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]]
    rmask = img[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]]

    y_left, x_left, y_right, x_right = wire_pos(img)

    y_left_ROI = y_left - ROI[1][1]
    x_left_ROI = x_left - ROI[0][0]

    y_right_ROI = y_right - ROI[3][1]
    x_right_ROI = x_right - ROI[2][0]

    cimg = img.copy()

    ## ---- CHECKS POUR VERIFIER LE CABLAGE ----

    # 1. Vérifier si tous les fils sont présents : 
    with console.status(f"[bold blue]{"Décompte des fils..."}[/bold blue]", spinner = 'dots'):
        with open("ModulePictures/iref_trim_per_module_v2.json", "r") as f:
            data = json.load(f)
            n_expected = expected_wire_number(extract_serial_number(path),data)
            n_detected = len(y_left) + len(y_right)

    print_info("Décompte des fils effectué.")

    if n_expected != n_detected: 
        print_error(f"Nombre de fils incorrect : {n_detected}/{n_expected}")
        # TODO : dire où il manque des fils
        # TODO : commande pour forcer à poursuivre les calculs



    else :
        print_success(f"Bon nombre de fils : {n_detected}/{n_expected}") 
        with console.status(f"[bold blue]{"Recherche des courts-circuits..."}[/bold blue]", spinner = 'dots'):


            short_list_left, endpoints_left, labels_left = find_shorts(lmask, 'left', y_left_ROI, x_left_ROI, draw)
            short_list_right, endpoints_right, labels_right = find_shorts(rmask, 'right', y_right_ROI, x_right_ROI, draw)

            print_info("Décompte des courts-circuits effectué.")

            if len(short_list_left) > 0 : print_error(f"{len(short_list_left)} potentiels courts-circuits à gauche.")
            else : print_success("Aucun court-circuit à gauche.")

            if len(short_list_right) > 0 : print_error(f"{len(short_list_right)} potentiels courts-circuits à droite.")
            else : print_success("Aucun court-circuit à droite.")

        with console.status(f"[bold blue]{"Détection des pistes..."}[/bold blue]", spinner = 'dots'):
            tracks = find_tracks(path)
        print_info("Pistes détectées.")

        with console.status(f"[bold blue]{"Vérification du câblage pour les fils non court-circuités..."}[/bold blue]", spinner = 'dots'):
            serial_number = extract_serial_number(path)
            trim_nb = iref_trim(serial_number, data)
            bounding_map_trim(trim_nb)
            wires = bounding_map_pads_pistes() # wires[i] donne le (pad, piste) associés au fil i. 
            # Format pad : XYYY, X n° du GA, YYY n°du pad
            # Format piste : XYY, X n° du GA, YY n°de piste

            nb_wires_off_track = 0

            # 2. Vérifier si des fils qui ne se touchent pas sont bien câblés.
            for label,endpoint in endpoints_left.items():

                (endpoint_location, wire_idx) = endpoint

                if endpoint_location != None:

                    # Vérifier si les fils vont bien au bon endroit
                    _, track_idx = wires[wire_idx[0]]
                    in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), measureDist = True) >= -4)
                    if in_track:
                        cv.circle(cimg, (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), 4, (0, 255, 0), 2)
                    else:
                        nb_wires_off_track += 1
                        cv.circle(cimg, (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), 6, (255, 0, 0), 3)
                
                else:
                    # Coloriage des fils court-circuités.
                    _, track_idx = wires[wire_idx[0]]
                    i=0
                    while i < len(wire_idx) and wires[wire_idx[i]][1] == track_idx : 
                        i+=1
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    color = [0, 128*(i==len(wire_idx)), 255]
                    cimg[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = color
                    
                

            for label, endpoint in endpoints_right.items():
                (endpoint_location, wire_idx) = endpoint
                wire_idx = [idx + len(y_left) for idx in wire_idx]

                if endpoint_location != None:
                    # Vérifier si les fils vont bien au bon endroit
                    _, track_idx = wires[wire_idx[0]]
                    in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), measureDist = True) >=-4)
                    if in_track:
                        cv.circle(cimg, (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), 4, (0, 255, 0), 2)
                    else:
                        nb_wires_off_track += 1
                        cv.circle(cimg, (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), 6, (255, 0, 0), 3)

                else :
                    # Coloriage des fils court-circuités.
                    _, track_idx = wires[wire_idx[0]]
                    i=0
                    while i < len(wire_idx) and wires[wire_idx[i]][1] == track_idx : 
                        i+=1
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    color = [0, 128*(i==len(wire_idx)), 255]
                    cimg[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = color

            for track in tracks.values():
                if len(track)==2 : 
                    cv.rectangle(cimg,track[0], track[1], (255, 128, 0), 1)
                elif len(track)>0 :
                    cv.polylines(cimg,[np.array(track).reshape((-1,1,2))], True, (255, 128, 0), 1)

            for short in short_list_left: 
                cv.circle(cimg, (short[0] + ROI[0][0], short[1] + ROI[1][1]), 4, (0, 255, 0), 2)
            for short in short_list_right: 
                cv.circle(cimg, (short[0] + ROI[2][0], short[1] + ROI[3][1]), 4, (0, 255, 0), 2)

            print_info("Vérification du câblage effectuée.")

        if ok :
            print_success("Module correctement cablé")
        else :
            
            afficher_bilan(n_detected, len(short_list_left), len(short_list_right), nb_wires_off_track)
            magnifying_glass(cimg)
            # afficher les zones (1/2/3/4) des fils manquants s'il y en a
            # afficher les zones des fils mal branchés s'il y en a
            # afficher les zones des fils qui se touchent s'il y en a
            # afficher des images du cablage