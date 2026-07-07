import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

import json

## Fonction utile pour normaliser un vecteur
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

## Fonction pour trouver le nom exact d'une image, à partir d'une liste de sub-string possibles
def find_suffix (module, sub_strings, folder) :
    for f in os.listdir(folder):
        for sub in sub_strings :
            name = os.path.basename(f)
            if (module in name) and (sub in name) :
                return (name)

## Fonction pour trouver l'image non câblée à l'image câblée (ou l'inverse)
def trouver_la_paire(fichier:str, dossier:str) -> str :
    """Finds the image corresponding to a given input

    Arguments :

    fichier - str : the name of the file to look for.

    dossier - str : the folder under which the images are located.

    Returns : str : path to the matching file.
    """

    with open("../config/config.json", "r") as f:
            config = json.load(f)

    bname=os.path.basename(fichier)
    after_bonding = config["suffix_after_bonding"]
    before_bonding = config["suffix_before_bonding"]

    first_ = bname.find("_")
    second_ = bname[first_:].find("_")
    name = bname[:second_]

    after_in_name = False
    before_in_name = False

    for a in after_bonding :
        if a in bname :
            after_in_name = True

    for b in before_bonding :
        if b in bname :
            before_in_name = True

    if "Ref_img" in bname :
        if "unbonded" in bname:
            return os.path.join("reference", "Ref_img_bonded.jpg")
        else : 
            return os.path.join("reference", "Ref_img_unbonded.jpg")

    elif after_in_name:
        for f in os.listdir(dossier):
            for b in before_bonding :
                if (b in os.path.basename(f)) and (name in os.path.basename(f)):
                    return os.path.join(dossier, f)
    elif before_in_name:
        for f in os.listdir(dossier):
            for a in after_bonding :
                if (a in os.path.basename(f)) and (name in os.path.basename(f)):
                    return os.path.join(dossier, f)
    return "Pas de paire"

#fonction utile pour afficher une image
def afficher(img) :
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

## Fonction qui affiche une liste de points sur une image.
## Paramètre with_cv : si on affiche avec opencv ou avec matplotlib
def afficher_points(img, centres, with_cv = False):
    img_copy = img.copy()
    for centre in centres :
        cv.circle(img_copy, (centre[0],centre[1]),15,(255,0,0),15)

    if with_cv :
        cv.imshow("Image", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        plt.imshow(img_copy)
        plt.show()