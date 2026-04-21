from Programs.output import *
from Programs.count import extract_serial_number, iref_trim
from Programs.map import bounding_map_pads_pistes, bounding_map_without_trim, bounding_map_trim
from Programs.wire_detection import *
import json

bounding_map_without_trim()

def run_check (path, iref = None, draw = False, verbose=0, config={}) :
    """
    Runs the checks for a given image.

    Arguments:
    path - str: path of the image to analyse
    draw - bool: whether the funciton should show additional information regarding the steps of the algorithm
    """

    with open("Configuration/config.json", "r") as f:
            config = json.load(f)
    lang = config["language"]
    if not verbose:
        verbose = config['verbose']

    console.print("\n")

    img = cv.imread(path)
    ROI = find_ROI(img, verbose_lv = verbose)
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
        with open("Reference/iref_trim_per_module_v2.json", "r") as f:
            data = json.load(f)
            n_expected = expected_wire_number(extract_serial_number(path),data, iref_=iref)
            n_detected = len(y_left) + len(y_right)

    if lang == "en" :
        print_info("Wire count completed.")
    else :
        print_info("Décompte des fils effectué.")

    if n_expected != n_detected: 
        if lang == "en" :
            print_error(f"Incorrect number of wires : {n_detected}/{n_expected}")
        else :
            print_error(f"Nombre de fils incorrect : {n_detected}/{n_expected}")
        # TODO : dire où il manque des fils
        # TODO : commande pour forcer à poursuivre les calculs



    else :
        if lang == "en" :
            print_success(f"Correct number of wires : {n_detected}/{n_expected}")
        else :
            print_success(f"Bon nombre de fils : {n_detected}/{n_expected}")
        
        running_message = "Recherche des courts-circuits..."
        if lang == "en" :
            running_message = "Short circuit detection..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):


            short_list_left, endpoints_left, labels_left = find_shorts(lmask, 'left', y_left_ROI, x_left_ROI, draw, verbose_lv=verbose, config=config)
            short_list_right, endpoints_right, labels_right = find_shorts(rmask, 'right', y_right_ROI, x_right_ROI, draw, verbose_lv=verbose, config=config)

            if lang == "en" :
                print_info("Short circuit count completed.")
            else :
                print_info("Décompte des courts-circuits effectué.")

            if len(short_list_left) > 0 :
                if lang == "en" :
                    print_error(f"{len(short_list_left)} potential short circuits on the left.")
                else : 
                    print_error(f"{len(short_list_left)} potentiels courts-circuits à gauche.")
            else : 
                if lang == "en" :
                    print_success("No short circuit on the left.")
                else : 
                    print_success("Aucun court-circuit à gauche.")

            if len(short_list_right) > 0 : 
                if lang == "en" :
                    print_error(f"{len(short_list_right)} potential short circuits on the right.")
                else :
                    print_error(f"{len(short_list_right)} potentiels courts-circuits à droite.")
            else : 
                if lang == "en" :
                    print_success("No short circuit on the right.")
                else : 
                    print_success("Aucun court-circuit à droite.")

        running_message = "Détection des pistes..."
        if lang == "en" :
            running_message = "Track detection..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):
            tracks = find_tracks(path, verbose_lv = verbose)
        if lang == "en" :
            print_info("Tracks detected.")
        else : 
            print_info("Pistes détectées.")

        running_message = "Vérification du câblage pour les fils non court-circuités..."
        if lang == "en" :
            running_message = "Checking the wiring for wires that are not short-circuited..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):
            serial_number = extract_serial_number(path)
            if iref == None :
                iref = iref_trim(serial_number, data)
            bounding_map_trim(iref)
            wires = bounding_map_pads_pistes() # wires[i] donne le (pad, piste) associés au fil i. 
            # Format pad : XYYY, X n° du GA, YYY n°du pad
            # Format piste : XYY, X n° du GA, YY n°de piste

            nb_wires_off_track = 0
            nb_crit_shorts_left = 0
            nb_not_crit_shorts_left = 0
            nb_crit_shorts_right = 0
            nb_not_crit_shorts_right = 0

            crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            non_crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            non_crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pads_mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # 2. Vérifier si des fils qui ne se touchent pas sont bien câblés.
            for label,endpoint in endpoints_left.items():

                (endpoint_location, wire_idx) = endpoint

                if endpoint_location != None:
                    # Vérifier si les fils vont bien au bon endroit
                    _, track_idx = wires[wire_idx[0]]
                    if (track_idx == 230) or (track_idx == 231) : 
                        endpoint_location = (endpoint_location[0] - 25, endpoint_location[1]) 
                    in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), measureDist = True) >= -4)
                    if in_track:
                        cv.circle(non_crit_endpoints_mask, (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), 4, 1, 2)
                    else:
                        nb_wires_off_track += 1
                        cv.circle(crit_endpoints_mask, (endpoint_location[0] + ROI[0][0], endpoint_location[1] + ROI[1][1]), 6, 1, 3)
                
                else:
                    # Coloriage des fils court-circuités.
                    _, track_idx = wires[wire_idx[0]]
                    i=0
                    while i < len(wire_idx) and wires[wire_idx[i]][1] == track_idx : 
                        i+=1
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    if i != len(wire_idx) : 
                        nb_crit_shorts_left += 1
                        crit_shorts_mask[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = 1
                    else : 
                        nb_not_crit_shorts_left += 1
                        non_crit_shorts_mask[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = 1
                    
                    
                

            for label, endpoint in endpoints_right.items():
                (endpoint_location, wire_idx) = endpoint
                wire_idx = [idx + len(y_left) for idx in wire_idx]

                if endpoint_location != None:
                    # Vérifier si les fils vont bien au bon endroit
                    _, track_idx = wires[wire_idx[0]]
                    if (track_idx == 332) or (track_idx == 333) : 
                        endpoint_location = (endpoint_location[0] + 25, endpoint_location[1]) 
                    in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), measureDist = True) >=-4)
                    if in_track:
                        cv.circle(non_crit_endpoints_mask, (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), 4, 1, 2)
                    else:
                        nb_wires_off_track += 1
                        cv.circle(crit_endpoints_mask, (endpoint_location[0] + ROI[2][0], endpoint_location[1] + ROI[3][1]), 6, (255, 0, 0), 3)

                else :
                    # Coloriage des fils court-circuités.
                    _, track_idx = wires[wire_idx[0]]
                    i=0
                    while i < len(wire_idx) and wires[wire_idx[i]][1] == track_idx : 
                        i+=1
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    if i != len(wire_idx) : 
                        nb_crit_shorts_right += 1 
                        crit_shorts_mask[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = 1
                    else : 
                        nb_not_crit_shorts_right += 1
                        non_crit_shorts_mask[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = 1

            for track in tracks.values():
                if len(track)>0 :
                    cv.polylines(pads_mask,[np.array(track).reshape((-1,1,2))], True, 1, 1)

            # for short in short_list_left: 
            #     cv.circle(cimg, (short[0] + ROI[0][0], short[1] + ROI[1][1]), 4, (0, 255, 0), 2)
            # for short in short_list_right: 
            #     cv.circle(cimg, (short[0] + ROI[2][0], short[1] + ROI[3][1]), 4, (0, 255, 0), 2)

            if lang == "en" :
                print_info("Wiring check completed.")
            else : 
                print_info("Vérification du câblage effectuée.")

        with open("Temp/path.txt", "w") as f:
            f.write(path)

        np.save("Temp/crit_shorts_mask.npy", crit_shorts_mask)
        np.save("Temp/non_crit_shorts_mask.npy", non_crit_shorts_mask)
        np.save("Temp/crit_endpoints_mask.npy", crit_endpoints_mask)
        np.save("Temp/non_crit_endpoints_mask.npy", non_crit_endpoints_mask)
        np.save("Temp/pads_mask.npy", pads_mask)

        afficher_bilan(n_detected, nb_not_crit_shorts_left, nb_not_crit_shorts_right, nb_crit_shorts_left, nb_crit_shorts_right, nb_wires_off_track)
        magnifying_glass_final_result(cimg)