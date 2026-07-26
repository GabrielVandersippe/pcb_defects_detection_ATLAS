from programs.output import *
from programs.count import extract_serial_number, iref_trim
from programs.map import bounding_map_pads_pistes, bounding_map_without_trim, bounding_map_trim
from programs.wire_detection import *
from programs.pads import find_pads
import json
import copy

bounding_map_without_trim()

def run_check (path, iref = None, draw = False, verbose=0, config={}, override_targets=False, full_image_mode = False, skip_pulltest=False, read_corners=False, read_targets=False) :
    """
    Runs the checks for a given image.

    Arguments:
    path - str: path of the image to analyse
    draw - bool: whether the function should show additional information regarding the steps of the algorithm
    """

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
        with open("reference/iref_trim_per_module_v2.json", "r") as f:
            data = json.load(f)
            n_expected = expected_wire_number(extract_serial_number(path),data, iref_=iref)
            if skip_pulltest :
                n_expected -= 25
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
    
    else :
        if lang == "en" :
            print_success(f"Correct number of wires : {n_detected}/{n_expected}")
        else :
            print_success(f"Bon nombre de fils : {n_detected}/{n_expected}")

    while True:

        running_message = "Recherche des courts-circuits..."
        if lang == "en" :
            running_message = "Short circuit detection..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):

            short_list_left, endpoints_left_pcb, endpoints_left_chip, labels_left = find_shorts(lmask, 'left', y_left_ROI, x_left_ROI, draw=False, verbose_lv=verbose, config=config)
            short_list_right, endpoints_right_pcb, endpoints_right_chip, labels_right = find_shorts(rmask, 'right', y_right_ROI, x_right_ROI, draw=False, verbose_lv=verbose, config=config)

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
        
        running_message = "Détection des pads..."
        if lang == "en" :
            running_message = "Pad detection..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):
            pads = find_pads(path, draw = False, verbose = verbose, config=config, read_corners=read_corners, full_image_mode=full_image_mode)
        if lang == "en" :
            print_info("Pads detected.")
        else : 
            print_info("Pads détectés.")

        running_message = "Détection des pistes..."
        if lang == "en" :
            running_message = "Track detection..."
        
        with console.status(f"[bold blue]{running_message}[/bold blue]", spinner = 'dots'):
            tracks = find_tracks(path, verbose_lv = verbose, override_targets=override_targets, full_image_mode = full_image_mode, read_targets=read_targets, config=config)
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
            map1, map2, map3, map4 = bounding_map_trim(iref)
            if skip_pulltest :
                map1[152:162] = 0
                map2[155:160] = 0
                map3[150:160] = 0
            wires = bounding_map_pads_pistes() # wires[i] donne le (pad, piste) associés au fil i. 
            # Format pad : XYYY, X n° du GA, YYY n°du pad
            # Format piste : XYY, X n° du GA, YY n°de piste

            nb_wires_off_track = 0
            nb_wires_off_pad = 0
            nb_crit_shorts_left = 0
            nb_not_crit_shorts_left = 0
            nb_crit_shorts_right = 0
            nb_not_crit_shorts_right = 0

            crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            non_crit_shorts_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            non_crit_endpoints_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pads_mask = np.zeros(img.shape[:2], dtype=np.uint8)

            list_missing = []
            missing_in_a_row = []
            list_crit_short_left = []
            list_non_crit_short_left = []
            list_crit_short_right = []
            list_non_crit_short_right = []
            list_crit_track = []
            list_crit_pad = []

            # 2. Vérifier si des fils qui ne se touchent pas sont bien câblés.
            iref_read = [([False] * 4) for i in range (4)] # iref_read[0] is GA1 / iref_read[1] is GA2 / iref_read[2] is GA3 / iref_read[3] is GA4
            last_pad = 1001
            pad_idx = 1001
            for label,endpoint in endpoints_left_pcb.items():

                (endpoint_location_pcb, wire_idx_pcb) = endpoint
                (endpoint_location_chip, wire_idx_chip) = endpoints_left_chip[label]

                expected_pad_idx, _ = wires[wire_idx_pcb[0] + len(list_missing)]

                if endpoint_location_pcb != None:
                    # Vérifier si les fils vont bien au bon endroit
                    in_pad = (cv.pointPolygonTest(pads[last_pad], (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), measureDist = True) >= -1) #temp
                    while not(in_pad) and (last_pad < 2199) :
                        last_pad += 1
                        if last_pad == 1199 :
                            last_pad = 2001
                        if last_pad != 2199 : 
                            in_pad = (cv.pointPolygonTest(pads[last_pad], (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), measureDist = True) >= -1) #temp
                    if last_pad == 2199 : # Si on atteint le bout de la liste de pads, le fil n'est pas bien cablé sur le chip
                        last_pad = pad_idx
                        nb_wires_off_pad += 1
                        list_crit_pad.append(expected_pad_idx)
                        cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 6, 1, 3)
                    else :
                        pad_idx = last_pad
                        if (pad_idx > expected_pad_idx) and (n_detected != n_expected) :
                            list_missing.append(wires[wire_idx_pcb[0] + len(list_missing) - len(missing_in_a_row)][0])
                            missing_in_a_row.append(wires[wire_idx_pcb[0] + len(list_missing) - len(missing_in_a_row)][0])
                        else :
                            missing_in_a_row = []
                        if (pad_idx != expected_pad_idx) and (n_detected == n_expected) :
                            nb_wires_off_pad += 1
                            list_crit_pad.append(expected_pad_idx)
                            cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 6, 1, 3)
                            
                        
                        off_pad = False
                        if pad_idx < 2000 :
                            track_idx = map1[pad_idx - 1001] + 100
                            if track_idx == 100 : # Si le fil est détecté sur un pad où il ne devrait pas y avoir de fil
                                off_pad = True
                                nb_wires_off_pad += 1
                                list_crit_pad.append(pad_idx)
                                cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 6, 1, 3)
                            else :
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 4, 1, 2)
                        else :
                            track_idx = map2[pad_idx - 2001] + 200
                            if track_idx == 200 : # Si le fil est détecté sur un pad où il ne devrait pas y avoir de fil
                                off_pad = True
                                nb_wires_off_pad += 1
                                list_crit_pad.append(pad_idx)
                                cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 6, 1, 3)
                            else :
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_chip[0] + ROI[0][0], endpoint_location_chip[1] + ROI[1][1]), 4, 1, 2)
                        if (track_idx == 230) or (track_idx == 231) : 
                            endpoint_location_pcb = (endpoint_location_pcb[0] - 25, endpoint_location_pcb[1]) 
                        if not(off_pad) :
                            in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location_pcb[0] + ROI[0][0], endpoint_location_pcb[1] + ROI[1][1]), measureDist = True) >= -4)
                            if in_track:
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_pcb[0] + ROI[0][0], endpoint_location_pcb[1] + ROI[1][1]), 4, 1, 2)
                            else:
                                nb_wires_off_track += 1
                                cv.circle(crit_endpoints_mask, (endpoint_location_pcb[0] + ROI[0][0], endpoint_location_pcb[1] + ROI[1][1]), 6, 1, 3)
                                list_crit_track.append(pad_idx)
                    
                    # Check if the wire's endpoint on the chip is in a iref pad
                    if (pad_idx == 1047) :
                        iref_read[0][0] = True
                    elif (pad_idx == 1048) :
                        iref_read[0][1] = True
                    elif (pad_idx == 1049) :
                        iref_read[0][2] = True
                    elif (pad_idx == 1050) :
                        iref_read[0][3] = True
                    elif (pad_idx == 2047) :
                        iref_read[1][0] = True
                    elif (pad_idx == 2048) :
                        iref_read[1][1] = True
                    elif (pad_idx == 2049) :
                        iref_read[1][2] = True
                    elif (pad_idx == 2050) :
                        iref_read[1][3] = True


                else:
                    # Coloriage des fils court-circuités.
                    pad_idx0, track_idx0 = wires[wire_idx_pcb[0] + len(list_missing)]
                    pad_idx_list = [pad_idx0]
                    crit = False
                    for i in range(len(wire_idx_pcb)) :
                        pad_idx, track_idx = wires[wire_idx_pcb[i] + len(list_missing)]
                        pad_idx_list.append(pad_idx)
                        if track_idx != track_idx0 :
                            crit = True
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    if crit : 
                        nb_crit_shorts_left += 1
                        list_crit_short_left += pad_idx_list
                        crit_shorts_mask[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = 1
                    else : 
                        nb_not_crit_shorts_left += 1
                        list_non_crit_short_left += pad_idx_list
                        non_crit_shorts_mask[ROI[1][1]:ROI[1][0], ROI[0][0]:ROI[0][1]][labels_left == label] = 1       
                

            last_pad = 3001
            pad_idx = 3001
            for label, endpoint in endpoints_right_pcb.items():

                (endpoint_location_pcb, wire_idx_pcb) = endpoint
                (endpoint_location_chip, wire_idx_chip) = endpoints_right_chip[label]
                wire_idx_pcb = [idx + len(y_left) for idx in wire_idx_pcb]

                expected_pad_idx, _ = wires[wire_idx_pcb[0] + len(list_missing)]

                if endpoint_location_pcb != None:
                    # Vérifier si les fils vont bien au bon endroit
                    in_pad = (cv.pointPolygonTest(pads[last_pad], (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), measureDist = True) >= -1)
                    while not(in_pad) and (last_pad < 4199) :
                        last_pad += 1
                        if last_pad == 3199 :
                            last_pad = 4001
                        # cas 4199 : (sur le premier fil !) reprendre pad_idx et dire qu'un fil n'a pas pu être lu...
                        if last_pad != 4199 :
                            in_pad = (cv.pointPolygonTest(pads[last_pad], (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), measureDist = True) >= -1)
                    if last_pad == 4199 : # Si on atteint le bout de la liste de pads, le fil n'est pas bien cablé sur le chip
                        last_pad = pad_idx
                        nb_wires_off_pad += 1
                        list_crit_pad.append(pad_idx)
                        cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 6, (255, 0, 0), 3)
                    else :
                        pad_idx = last_pad
                        if (pad_idx > expected_pad_idx) and (n_detected != n_expected) : # vrai fil manquant
                                list_missing.append(wires[wire_idx_pcb[0] + len(list_missing) - len(missing_in_a_row)][0])
                                missing_in_a_row.append(wires[wire_idx_pcb[0] + len(list_missing) - len(missing_in_a_row)][0])
                        else :
                            missing_in_a_row = []
                        if (pad_idx != expected_pad_idx) and (n_detected == n_expected) : # fil cablé sur un mauvais pad
                            print(pad_idx)
                            print(expected_pad_idx)

                            nb_wires_off_pad += 1
                            list_crit_pad.append(pad_idx)
                            cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 6, (255, 0, 0), 3)
                        off_pad = False
                        if pad_idx < 4000 :
                            track_idx = map3[pad_idx - 3001] + 300
                            if track_idx == 300 : # Si le fil est détecté sur un pad où il ne devrait pas y avoir de fil
                                off_pad = True
                                nb_wires_off_pad += 1
                                list_crit_pad.append(pad_idx)
                                cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 6, (255, 0, 0), 3)
                            else :
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 4, 1, 2)
                        else :
                            track_idx = map4[pad_idx - 4001] + 400
                            if track_idx == 400 : # Si le fil est détecté sur un pad où il ne devrait pas y avoir de fil
                                off_pad = True
                                nb_wires_off_pad += 1
                                list_crit_pad.append(pad_idx)
                                cv.circle(crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 6, (255, 0, 0), 3)
                            else :
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_chip[0] + ROI[2][0], endpoint_location_chip[1] + ROI[3][1]), 4, 1, 2)
                        if (track_idx == 332) or (track_idx == 333) : 
                            endpoint_location_pcb = (endpoint_location_pcb[0] + 25, endpoint_location_pcb[1]) 
                        if not(off_pad) :
                            in_track = (cv.pointPolygonTest(tracks[track_idx], (endpoint_location_pcb[0] + ROI[2][0], endpoint_location_pcb[1] + ROI[3][1]), measureDist = True) >=-4)

                            if in_track:
                                cv.circle(non_crit_endpoints_mask, (endpoint_location_pcb[0] + ROI[2][0], endpoint_location_pcb[1] + ROI[3][1]), 4, 1, 2)
                            else:
                                nb_wires_off_track += 1
                                list_crit_track.append(pad_idx)
                                cv.circle(crit_endpoints_mask, (endpoint_location_pcb[0] + ROI[2][0], endpoint_location_pcb[1] + ROI[3][1]), 6, (255, 0, 0), 3)
                        

                    # Check if the wire's endpoint on the chip is in a iref pad
                    if (pad_idx == 3047) :
                        iref_read[2][0] = True
                    elif (pad_idx == 3048) :
                        iref_read[2][1] = True
                    elif (pad_idx == 3049) :
                        iref_read[2][2] = True
                    elif (pad_idx == 3050) :
                        iref_read[2][3] = True
                    elif (pad_idx == 4047) :
                        iref_read[3][0] = True
                    elif (pad_idx == 4048) :
                        iref_read[3][1] = True
                    elif (pad_idx == 4049) :
                        iref_read[3][2] = True
                    elif (pad_idx == 4050) :
                        iref_read[3][3] = True


                else :
                    # Coloriage des fils court-circuités.
                    pad_idx0, track_idx0 = wires[wire_idx_pcb[0] + len(list_missing)]
                    pad_idx_list = [pad_idx0]
                    crit = False
                    for i in range(len(wire_idx_pcb)) :
                        pad_idx, track_idx = wires[wire_idx_pcb[i] + len(list_missing)]
                        pad_idx_list.append(pad_idx)
                        if track_idx != track_idx0 :
                            crit = True
                    # Mettre le fil en rouge, ou en orange s'ils arrivent tous sur une même piste.
                    if crit : 
                        nb_crit_shorts_right += 1
                        list_crit_short_right += pad_idx_list 
                        crit_shorts_mask[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = 1
                    else : 
                        nb_not_crit_shorts_right += 1
                        list_non_crit_short_right += pad_idx_list
                        non_crit_shorts_mask[ROI[3][1]:ROI[3][0], ROI[2][0]:ROI[2][1]][labels_right == label] = 1

        # Checks if there are lots of errors: probably false positives, maybe due to a change in lighting compared to the reference.
        if nb_wires_off_track >= 200 or nb_wires_off_pad>=200:
            if lang=='fr': 
                console.print(f"[bold dark_orange]ATTENTION : [/bold dark_orange][dark_orange]Beacoup d'erreurs de câblage ont été détectées. Cela peut être dû à un changement d'éclairage entre l'image et les références.")
                console.print(f"[dark_orange]Recommendation : Ceci pourrait être réglé en réduisant l'aggressivité du programme. Voulez-vous le faire ?")
                user_input = console.input("[bold dark_orange][Y/N] : [/bold dark_orange]")
            else: 
                console.print(f"[bold dark_orange]WARNING: [/bold dark_orange][dark_orange]Many errors have been detected. This may be due to a change in lighting between the image and the references..")
                console.print(f"[dark_orange]Recommendation: This maybe fixed by lowering the aggressiveness of the program. Proceed?")
                user_input = console.input("[bold dark_orange][Y/N]: [/bold dark_orange]")

            if user_input.strip().upper() == 'Y':

                with open("../config/aggressiveness_config.json", 'r') as f:
                    full_config = json.load(f)
                    
                custom_config = copy.deepcopy(full_config[config["aggressiveness_level"]])
                
                custom_config["wire_threshold"]["satmask_thr"]["ub"] += 10
                custom_config["wire_threshold"]["valmask_thr"]["lb"] -= 30
                
                full_config["custom"] = custom_config
                
                with open("../config/aggressiveness_config.json", "w") as f:
                    json.dump(full_config, f, indent=4)
                    
                config["aggressiveness_level"] = "custom"

                continue

        # Computes iref to get a number
        iref_nb_read = []
        iref_map = [[True, True, True, True], [False, True, True, True], [True, False, True, True], [False, False, True, True], [True, True, False, True], [False, True, False, True], [True, False, False, True], [False, False, False, True], [True, True, True, False], [False, True, True, False], [True, False, True, False], [False, False, True, False], [True, True, False, False], [False, True, False, False], [True, False, False, False], [False, False, False, False]]
        for i in range(len(iref_read)) :
            for j in range (len(iref_map)) :
                if iref_read[i] == iref_map[j] :
                    iref_nb_read.append(j)
            if iref_nb_read[i] == iref[i] :
                if lang == "en" :
                    print_success("Correct IREF for GA"+str(i+1))
                    print_success("IREF read : " + str(iref_nb_read[i]))
                    print_success("IREF expected : " + str(iref[i]))
                else :
                    print_success("Bon IREF pour GA"+str(i+1))
                    print_success("IREF lu : " + str(iref_nb_read[i]))
                    print_success("IREF attendu : " + str(iref[i]))
            else :
                if lang == "en" :
                    print_error("False IREF for GA"+str(i+1))
                    print_error("IREF read : " + str(iref_nb_read[i]))
                    print_error("IREF expected : " + str(iref[i]))
                else :
                    print_error("Faux IREF pour GA"+str(i+1))
                    print_error("IREF lu : " + str(iref_nb_read[i]))
                    print_error("IREF attendu : " + str(iref[i]))


        for track in tracks.values():
            if len(track)>0 :
                cv.polylines(pads_mask,[np.array(track).reshape((-1,1,2))], True, 1, 1)
        
        for pad in pads.values():
            if len(pad)>0 :
                cv.polylines(pads_mask,[np.array(pad).reshape((-1,1,2))], True, 1, 1)


        if lang == "en" :
            print_info("Wiring check completed.")
        else : 
            print_info("Vérification du câblage effectuée.")
        
        break

    with open("../temp/path.txt", "w") as f:
        f.write(path)

    np.save("../temp/crit_shorts_mask.npy", crit_shorts_mask)
    np.save("../temp/non_crit_shorts_mask.npy", non_crit_shorts_mask)
    np.save("../temp/crit_endpoints_mask.npy", crit_endpoints_mask)
    np.save("../temp/non_crit_endpoints_mask.npy", non_crit_endpoints_mask)
    np.save("../temp/pads_mask.npy", pads_mask)

    afficher_bilan(n_detected, n_expected, nb_not_crit_shorts_left, nb_not_crit_shorts_right, nb_crit_shorts_left, nb_crit_shorts_right, nb_wires_off_track, nb_wires_off_pad, iref, iref_nb_read, list_missing, list_crit_short_left, list_non_crit_short_left, list_crit_short_right, list_non_crit_short_right, list_crit_track, list_crit_pad)
    if path[-1] == "/" :
        file_name = (path.split("/")[-2]).split(".")[0]
    else :
        file_name = (path.split("/")[-1]).split(".")[0]
    magnifying_glass_final_result(cimg, file_name)
