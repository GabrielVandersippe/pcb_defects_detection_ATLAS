from Programs.output import print_success, print_error, print_info
from Programs.count import extract_serial_number, iref_trim
from Programs.map import bounding_map_pads_pistes, bounding_map_without_trim, bounding_map_trim
import json
import numpy as np

with open("ModulePictures/iref_trim_per_module_v2.json", "r") as f:
    data = json.load(f)

bounding_map_without_trim()

def run_check (file_name) :
    serial_number = extract_serial_number(file_name)
    trim_nb = iref_trim(serial_number, data)

    bounding_map_trim(trim_nb)
    wires = bounding_map_pads_pistes()
    nb_wires_wanted = len(wires)

    ok = True
    # ajouter le traitement de l'image pour avoir le nombre de fils réels et leurs connections
    
    if ok :
        print_success("Module correctement cablé")
    else :
        print_error("Module incorrect")
        print_info("Nombre des fils : " + "/" + str(nb_wires_wanted)) # insérer le nombre de fils comptés
        # afficher les zones (1/2/3/4) des fils manquants s'il y en a
        # afficher les zones des fils mal branchés s'il y en a
        # afficher les zones des fils qui se touchent s'il y en a
        # afficher des images du cablage