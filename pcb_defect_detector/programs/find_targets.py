import cv2 as cv
import numpy as np
from programs.utils import *
from programs.data import *
from programs.output import console

import json


def unique_centers(centers, idx, nbmires, min_dist = 30): #distance minimale entre deux centres
    """
    Remove centers that are too close to each other.
    
    Arguments :

    centers: list of (x, y) tuples
    min_dist: minimum allowed distance between centers
    idx : list of sorted indexes for centers
    nbmires : number of targets to find

    Returns : list of unique centers
    """
    unique_centers = []
    i = 0
    while len(unique_centers)<nbmires and i<len(idx):
        current_center = centers[idx[i]]

        if all(np.hypot(c[0]-current_center[0], c[1]-current_center[1]) > min_dist for c in unique_centers):
            unique_centers.append(current_center)

        i+=1
    return unique_centers





def mires_template_matching(img_input:np.ndarray, draw = False, verbose_lv=0, **kwargs):
    """Finds the positions of the 8 targets on the unwired PCB, or an error if it could not.

    Arguments :

    img_input - array of pixels : the working image (in BGR)

    draw - bool : whether or not the function should return images of what it is doing.

    Returns : np.ndarray : array of centers
    """
    assert img_input is not None, "Le fichier n'a pas pu être lu, vérifier avec os.path.exists()" #Vérifier si l'image existe

    template = cv.imread("reference/Template_Thresh_cropped.png", cv.IMREAD_GRAYSCALE)

    img_name = kwargs.get('img_name')
    if verbose_lv>2 and img_name : console.log(f"Détection des mires sur l'image : {img_name}.")

    # Preprocess
    gray = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    img = cv.medianBlur(gray, 5)

    (height, length) = gray.shape

    ## TODO : changer cette extraction à la main par quelque chose d'automatisé
    sliceparams = [(100, 400, 300, 600, 1),  
                   (-400, -100, 300, 600, 1),
                   (3800, 4200, 500, 800, 2), 
                   (100, 400, -600,-300, 1), 
                   (-400, -100, -600, -300, 1), 
                   (3800, 4200, -800, -500, 2)]

    # cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR) #Avec 3 canaux pour pouvoir l'afficher bien

    centers = []

    for (beg1, end1, beg2, end2, nbmires) in sliceparams :
        mask = img[beg1:end1, beg2:end2]

        #Threshold adaptif
        thr = cv.adaptiveThreshold(
            mask, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            31, 7
        )

        res = cv.matchTemplate(thr,template,cv.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where( res >= threshold)

        slice_centers = []

        for pt in zip(*loc[::-1]):
            center = (pt[0]+beg2%length + 28, pt[1]+beg1%height + 28) ##NB : Coordonnees du centre dans le template : (28,28)
            slice_centers.append(center)
            # if draw:
            #     cv.circle(cimg, center, 40, (0,0,255), 3)
            #     cv.circle(cimg,center,2,(0,0,255),3)
        
        #S'il y a plus ou autant de mires que desire, on les regarde quand meme pour choisir les meilleurs matchs et ne pas prendre de doublons
        # On prend les meilleurs matchs
        sorted_idx = np.argsort([res[pt[1], pt[0]] for pt in zip(*loc[::-1])])[::-1]
        slice_centers = unique_centers(slice_centers, sorted_idx, nbmires)
        
        if verbose_lv>2 : console.log(f"Slice [{beg1}:{end1}, {beg2}:{end2}] : {len(slice_centers)} mires trouvées sur {nbmires} par template matching.")
        # S'il y a moins de slices que prevu, on prévient qu'il y a une erreur 
        if len(slice_centers) < nbmires:

            # Option de secours : recherche de cercles sur le masque (Failsafe)
            circles = cv.HoughCircles(mask,cv.HOUGH_GRADIENT,1,minDist = 100,
                                    param1=300,param2=15,minRadius=15 ,maxRadius=30)
            if circles is not None:
                circles = np.int32(np.around(circles))
                for circle in circles[0,:nbmires]:
                    center = (circle[0]+beg2%length,circle[1]+beg1%height)
                    slice_centers.append(center)
                    # if draw:
                    #     cv.circle(cimg, center, circle[2], (0,0,255), 3)
                    #     cv.circle(cimg, center,2,(0,0,255),3)

            if verbose_lv>2 : console.log(f"{len(slice_centers)} mires trouvées sur {nbmires} par recherche de cercles.")

            # Si rien trouvé, on renvoie un warning
        if len(slice_centers) < nbmires:
            print(f"Attention : devait trouver {nbmires} mires, n'en a trouvé que {len(slice_centers)}. Les résultats peuvent être moins précis. Le problème est affiché dans une autre fenêtre.")
            cv.imshow("Problème", mask)
            cv.waitKey(0)
            cv.destroyAllWindows()

            if len(slice_centers) == 0:
                slice_centers = [(-1,-1) for _ in range(nbmires)]
            else :
                if slice_centers[0][1] < height//2 : #Si la mire sont dans la partie haute de l'image, on suppose que c'est la mire du haut qui a été trouvée
                    slice_centers += [(-1,-1)] #On met -1 à la fin de la liste pour la mire du bas
                else : 
                    slice_centers = [(-1,-1)] + slice_centers #On met -1 au début de la liste pour la mire du haut
        
        #Si on est ici, c'est qu'on en a peut être deux, on s'assure qu'elles sont bien ordonnées.
        if len(slice_centers)==2:
            slice_centers.sort(key=lambda x: x[1], reverse=False)

        centers.extend(slice_centers)

    # if draw :

    #     for i,(beg1, end1, beg2, end2, _) in enumerate(sliceparams):

    #         cv.imshow("Mire " + str(i), cimg[beg1:end1, beg2:end2])
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()

    return np.array(centers)



def compute_homography_center(uncabled_img, cabled_img, crop_ratio=0.6):
    """
    Calcule une homographie entre images cablees et non cablees par matching de la région centrale

    Arguments :

    crop_ratio = ratio hauteur/largeur conservée autour du centre (0.4 = 40%)
    """

    # On convertit en grayscale
    uncbld_gray = cv.cvtColor(uncabled_img, cv.COLOR_BGR2GRAY)
    cbld_gray = cv.cvtColor(cabled_img, cv.COLOR_BGR2GRAY)


    # Extraire la region centrale qu'on veut identifier sur les deux PCB
    def center_crop(img, ratio):
        h, w = img.shape
        kept_h, kept_w = int(h*ratio), int(w*ratio)
        y1, y2 = (h - kept_h)//2, (h + kept_h)//2
        x1, x2 = (w - kept_w)//2, (w + kept_w)//2
        return img[y1:y2, x1:x2], (x1, y1)

    uncbld_crop, offset1 = center_crop(uncbld_gray, crop_ratio)
    cbld_crop, offset2 = center_crop(cbld_gray, crop_ratio)

    sift = cv.SIFT_create(nfeatures=3000) #Les features que l'on veut trouver dans cette région centrale

    keypoints1, descriptors1 = sift.detectAndCompute(uncbld_crop, None)
    keypoints2, descriptors2 = sift.detectAndCompute(cbld_crop, None)

    #On fait matcher ces descriptions
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # on  conserve les bons matchs
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good) < 10:
        raise ValueError("Trop peu de correspondances pour calculer l'homographie.")

    #on fait les correspondances
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good])

    # Et pour finir on remet ces coordonnees dans l'image entiere non croppee
    pts1[:,0] += offset1[0]
    pts1[:,1] += offset1[1]
    pts2[:,0] += offset2[0]
    pts2[:,1] += offset2[1]

    # on peut alors enfin finir en trouvant l'homographie entre l'image non cablee et l'image cablee !!
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
    if H is None:
        raise ValueError("Échec du calcul de l'homographie.")
    
    return H



def warp_points(points, H):
    """
    Appliquer l'homographie à une liste de points
    """

    warped = []
    for point in points :
        if point[0] != -1:
            warped.append(np.int32(cv.perspectiveTransform(np.float32(point).reshape(-1,1,2), H).reshape(-1,2))[0])
        else:
            warped.append([-1,-1])
    return np.array(warped)



def find_targets_wired(path:str, draw=False, verbose_lv = 0, **kwargs):
    """
    Find the positions of the 8 targets on the wired PCB, or an error if it could not.

    Arguments :

    path - str : path to the image of the wired PCB

    Returns : np.ndarray : array of centers
    """
    with open("../config/config.json", "r") as f:
            config = json.load(f)

    cabled_img = cv.imread(path)
    uncabled_img = cv.imread(trouver_la_paire(path, config["pictures_folder"]))

    H = compute_homography_center(uncabled_img, cabled_img)

    unwired_centers = mires_template_matching(uncabled_img, draw, verbose_lv=verbose_lv, kwargs = kwargs)
    wired_centers = warp_points(unwired_centers, H).astype(np.int32)

    return wired_centers