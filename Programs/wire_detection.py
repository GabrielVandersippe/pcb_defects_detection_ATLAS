import cv2 as cv
import numpy as np
from Programs.utils import *
from Programs.data import *
from Programs.find_targets import *
from Programs.debug_tools import *
from Programs.wire import *
from Programs.count import *
from Programs.tracks import *



def kernel_wheel(shape, verbose_lv = 0):
    """
    Creates a list of kernels with a rotating diagonal, spanning diagonal to antidiagonal.

    Arguments:
    shape - tuple of int: the shape of the output kernel

    Returns:
    kernels - list of arrays
    """
    x, y = shape
    kernels = []
    
    for i in range(y // 2 + y % 2):
        height = y - 2 * i
            
        mat = np.zeros((height, x), dtype=np.uint8)
        
        for j in range(height):
            start_col = int(j * x / height)
            end_col = int(np.ceil((j + 1) * x / height))
            
            mat[j, start_col:end_col] = 1
            
        kernels.append(mat)
        
        if height > 1 or i == 0:
            kernels.append(np.flip(mat, axis=1))
    
    if verbose_lv > 2: console.log("Noyaux d'ouverture & fermeture générés.")
            
    return kernels



def find_ROI(img, verbose_lv = 0):
    """
    Finds the regions where the wires are located.

    Arguments:
    img - np.ndarray: input image

    Returns:
    List of points defining the left and right ROI respectively.
    """
    n = img.shape[1]

    high_left, low_left = crop_ligns(img[:,:n//2])
    high_right, low_right = crop_ligns(img[:,n//2:])
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    left = crop_columns_left(grey[high_left:low_left]) + 10 # On ne veut pas prendre en compte les pads.
    right = crop_columns_right(grey[high_right:low_right]) - 10

    if verbose_lv > 2: console.log("Région d'intérêt extraite.")

    return [(left, left + 500),(low_left, high_left), (right-500,right), (low_right,high_right)]



def wire_threshold(img, side, verbose_lv=0):
    """
    Extracts the shape of the wires using multiple thersholing methods and morphological operations

    Arguments:
    img - np.ndarray : input image
    side - 'left' or 'right': side where we are currently operating

    Returns:
    clean - np.ndarray : processed version of the image.
    """

    # Retire les portions les plus jaunes ou gris foncé de l'image

    total_pixels = img.shape[0]*img.shape[1]

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    _,S,V = cv.split(hsv)
    satmask = cv.inRange(S, 0, 60)
    valmask = cv.inRange(V, 180 - 20*(side=='right'),255)
    colormask = cv.bitwise_and(satmask,valmask)

    if verbose_lv>1 : console.log(f"Masque de saturation : {cv.countNonZero(satmask)} pixels blancs conservés sur {total_pixels} [{100*cv.countNonZero(satmask)/total_pixels}%]")
    if verbose_lv>1 : console.log(f"Masque de valeur : {cv.countNonZero(valmask)} pixels blancs conservés sur {total_pixels} [{100*cv.countNonZero(valmask)/total_pixels}%]")

    # Retire les parties les moins lumineuses de l'image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(5,5),0)
    intensitymask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25,-15)
    if verbose_lv>1 : console.log(f"Masque d'intensité en échelle de gris : {cv.countNonZero(intensitymask)} pixels blancs conservés sur {total_pixels} [{100*cv.countNonZero(intensitymask)/total_pixels}%]")

    # Resultat
    mask = cv.bitwise_and(intensitymask,colormask)

    # Nettoyage : passage d'un kernel horizontal
    closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,1))
    dilated = cv.morphologyEx(mask,cv.MORPH_DILATE, closing_kernel) # Fermeture

    kernels = kernel_wheel((35,9), verbose_lv = verbose_lv)  # Retrait des "blobs" a l'aide de noyaux customisés (diagonale "tournante")
    clean = cv.morphologyEx(dilated, cv.MORPH_OPEN, kernels[0])
    for kernel in kernels[1:]:
        clean = cv.bitwise_or(clean,cv.morphologyEx(dilated, cv.MORPH_OPEN, kernel))

    if verbose_lv>0 : console.log(f"Image nettoyée : {cv.countNonZero(clean)} pixels blancs conservés sur {total_pixels} [{100*cv.countNonZero(clean)/total_pixels}%]")


    return clean



def find_shorts(mask, input_side, y_left_list, x_left, draw = False, verbose_lv=0, **kwargs):
    """
    Finds the positions of the shorts, if there is any, and computes and approximated position for each unshorted wire.

    Arguments :
    mask - np.ndarray : the ROI of image that we want to analyse
    input_side - 'left' of 'right' : the side that is currently beign processed.
    y_left_list - array of int : the y position of the wires
    x_left - int : the x position of the wires
    draw - bool : whether the function should return visual aid regarding what it is doing

    Returns : 
    short_list - array of points : the list of shorts
    edge_dict - dict that maps a label to a (endpoint, index), where 'endpoint' is the ending point of a wire, and index is an array of one (or more, if shorted) indices that represent the wires.
    labels - mask containing every label, representing each the position of a wire.
    """
    side = input_side.lower()
    if side != 'left' and side != 'right':
        raise Exception("Mauvais argument pour 'side'. Entrer 'left' ou 'right'.")

    connectivity = 8 if 'connectivity' not in kwargs.keys() else kwargs['connectivity']
    processed = wire_threshold(mask, side, verbose_lv = verbose_lv)
    retval, labels, stats, _ = cv.connectedComponentsWithStats(processed, connectivity = connectivity)

    if verbose_lv>0 : console.log(f"{retval} composantes connexes détectées sur la partie {side} de l'image. [Connectivité : {connectivity}]")

    seen_labels = np.full(retval, False)

    if draw : 
        cimg = mask.copy()

    short_list = []
    edge_dict = {}

    for idx_wire, y_wire in enumerate(y_left_list):

        label = labels[y_wire, x_left]
           
        x = stats[label, cv.CC_STAT_LEFT]
        y = stats[label, cv.CC_STAT_TOP]
        w = stats[label, cv.CC_STAT_WIDTH]
        h = stats[label, cv.CC_STAT_HEIGHT]

        region = processed[y:y+h, x:x+w]*(labels[y:y+h, x:x+w] == label) #Extraire la region où il y a le fil
        region = cv.copyMakeBorder(region,5,5,5,5,cv.BORDER_CONSTANT, value = 0) #Rajouter des bords noirs sur les cotes pour pas avoir de mauvaises surprises


        if seen_labels[label]:  # On verifie qu'on ne voit qu'une fois chaque label, sinon c'est qu'il y a un court-circuit

            prev_idx = edge_dict[label][1]
            if verbose_lv>1 : console.log(f"Fil {idx_wire} court-circuité. [Liste des autres fils en court-circuit : {prev_idx}]")
            prev_idx.append(idx_wire)
            edge_dict[label] = (None, prev_idx)

            vkernel = cv.getStructuringElement(cv.MORPH_RECT, (1,7)) # Rectangle 1x7
            hkernel = cv.getStructuringElement(cv.MORPH_RECT, (25,3))
            no_wires = cv.morphologyEx(region, cv.MORPH_ERODE, vkernel) # Il ne reste que les endroits des courts-circuits (fil plus epais, et notamment plus large)
            shorts = cv.morphologyEx(no_wires, cv.MORPH_DILATE, hkernel) # Si plusieurs contacts dans la zone, on essaye de les relier pour minimiser le nombre de cercles

            if draw : 
                magnifying_glass(region)
                magnifying_glass(mask[y:y+h, x:x+w])

            nb_shorts,_,short_stats,short_centers = cv.connectedComponentsWithStats(shorts) #On trouve les emplacement des differents courts-circuits

            if verbose_lv>1 : console.log(f"Fil {idx_wire} : {nb_shorts} points de contact détéctés.")

            for i in range(nb_shorts):

                if side == 'left': is_not_beginning = short_stats[i, cv.CC_STAT_LEFT] > 15 # On vérifie si ce n'est pas a position où le pad est câblé (rappel : marge de 5px)
                else : is_not_beginning = short_stats[i, cv.CC_STAT_LEFT] + short_stats[i, cv.CC_STAT_WIDTH] < w  # + 10 - 10, ne doit pas être dans les 10 derniers pixels à droite


                if is_not_beginning :
                    short_list.append([int(short_centers[i][0] + x-5), int(short_centers[i][1] + y-5)])

                    if draw:
                        cx, cy = short_centers[i]
                        cv.circle(cimg, (int(cx+x-5), int(cy+y-5)), 20, (0,0,255), 3)   
        
        
        else :
            seen_labels[labels[y_wire, x_left]] = True # On verifie qu'on ne voit qu'une fois chaque label, sinon c'est qu'il y a un court-circuit
        
            # On trouve le point de soudure
            if side == 'left':
                right_pixels_y = np.where(region[:, -6] > 0)[0] # Tous les pixels blancs dans la colonne la plus a droite
                y_mean = int(np.mean(right_pixels_y))
                edgex = x + w - 1 + 3 # On dit que la soudure est un tout petit peu à droite (+3px ici)
                edgey = y-5 + y_mean

                end_region = region[:, -26:-6]
                wires = cv.HoughLines(end_region,1,np.pi/180, 0, 0)
                if wires is not None:
                    theta = wires[0][0][1]
                    # XXX Peut être fait sans avoir a calculer edgex et edgey
                    
                    solderx = edgex + 15
                    soldery = edgey - 15/np.tan(theta)
                    edge_dict[label] = ((int(solderx),int(soldery)), [idx_wire])
                else : 
                    edge_dict[label] = ((edgex,edgey), [idx_wire])
                
                if verbose_lv>2 : console.log(f"Soudure du fil {idx_wire} trouvée à la position {edge_dict[label]}.")
            
            else :
                left_pixels_y = np.where(region[:, 6] > 0)[0]
                y_mean = int(np.mean(left_pixels_y))
                edgex = x + 1 -3 # On dit que la soudure est un tout petit peu à gauche (-3px ici)
                edgey = y-5 + y_mean

                end_region = region[:, 6:26]
                wires = cv.HoughLines(end_region,1,np.pi/180, 0, 0)
                if wires is not None:
                    theta = wires[0][0][1]
                    # XXX Peut être fait sans avoir a calculer edgex et edgey
                    
                    solderx = edgex - 15
                    soldery = edgey + 15/np.tan(theta)
                    edge_dict[label] = ((int(solderx),int(soldery)), [idx_wire])
                else : 
                    edge_dict[label] = ((edgex,edgey), [idx_wire])
                
                if verbose_lv>2 : console.log(f"Soudure du fil {idx_wire} trouvée à la position {edge_dict[label]}.")

            if draw : 
                cv.circle(cimg, (int(edge_dict[label][0]), int(edge_dict[label][1])),4, (0, 255, 0), 2)

    if draw:
        magnifying_glass(cimg)

    return short_list, edge_dict, labels