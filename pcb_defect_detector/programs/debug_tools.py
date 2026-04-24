import cv2 as cv
from programs.output import console
from programs.utils import *
from programs.data import *
from programs.find_targets import *
from time import strftime, gmtime

import json

with open("../config/config.json", "r") as f:
        config = json.load(f)

drawing = False # true if mouse is pressed
ix,iy = -1,-1
patch_size=int(400/config["zoom"]) # 100 by default
zoom_scale=config["zoom"] # 4 by default
current_viewmode = 0 

def mouse_callback(event, x, y, flags, param):

    lang = config["language"]

    global ix,iy,drawing, patch_size, zoom_scale
    
    state = param
    img = state['current_view']

    # Define the ROI 
    x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
    x2, y2 = min(img.shape[1], x + patch_size // 2), min(img.shape[0], y + patch_size // 2)
    
    roi = img[y1:y2, x1:x2]
    
    if roi.size > 0:
        zoom_view = cv.resize(roi, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv.INTER_NEAREST)
        h_z, w_z = zoom_view.shape[:2]
        
        # -------------------------------------
        ## Ecriture des coordonnees sur la vue zoomée

        coord_text = f"X:{x} Y:{y}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        (text_w, text_h), _ = cv.getTextSize(coord_text, font, font_scale, thickness)
        # Calculate position (Bottom Right with 10px padding)
        text_x = w_z - text_w - 10
        text_y = h_z - 10
        cv.rectangle(zoom_view, (text_x - 5, text_y - text_h - 5), (w_z, h_z), (0, 0, 0), -1)
        cv.putText(zoom_view, coord_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv.LINE_AA)
        #------------------------------------
        
        if lang == "en" :
            cv.imshow("Magnifying glass", zoom_view)
        else :
            cv.imshow("Loupe", zoom_view)
    

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x_start, x_end = sorted([ix, x])
        y_start, y_end = sorted([iy, y])

        # Check if selected area large enough
        if (x_end - x_start) > 15 and (y_end - y_start) > 15:
            state['offset_x'] += x_start
            state['offset_y'] += y_start

            state['current_view'] = img[y_start:y_end, x_start:x_end]


def magnifying_glass(src):

    if type(src) == str:
        img = cv.imread(src)
    else:
        img = src

    state = {
        'current_view': img.copy(),
        'offset_x': 0,
        'offset_y': 0
    }

    cv.namedWindow("Vue Principale", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Vue Principale", mouse_callback, param=state)

    console.rule("[bold blue]LOUPE")
    console.print("")
    console.print("[blue]Passer la souris sur la vue principale pour afficher la version zoomée.")
    console.print("[blue][bold]Glisser-déposer[/bold] pour zoomer.")
    console.print("[blue]Appuyer sur [bold]'r'[/bold] pour réinitialiser la vue.")
    console.print("[blue]Appuyer sur [bold]'q'[/bold] pour quitter.")

    

    while True:
        cv.imshow("Vue Principale", state['current_view'])
        key = cv.waitKey(1) & 0xFF 
        
        if key == ord('q'): # Quitter
            break
        elif key == ord('r'): # Reset 
            state['current_view'] = img.copy()
            state['offset_x'], state['offset_y'] = 0, 0
        

    cv.destroyAllWindows()


def create_masked_view(img, mask, color):
    result = img.copy()
    darkened = cv.convertScaleAbs(result, alpha=0.3, beta=0)
    darkened[mask > 0] = color
    
    return darkened

def create_view_all_masks(img, masks, colors):
    result = img.copy()
    for i, mask in enumerate(masks):
        result[mask > 0] = colors[i]
    return result


def magnifying_glass_final_result(src):

    lang = config["language"]

    colors = [[0,0,255],    #red for crit shorts
              [0,128,255],  #orange for non crit, etc
              [0,255,0],    
              [255,0,0],
              [128,128,0]]

    if type(src) == str:
        img = cv.imread(src)
        assert img is not None, f"L'image d'origine n'a pas pu être lue, peut-être a-elle été déplacée. Le chemin devrait être {src}." if lang=='fr' else f"Image could not be read, maybe it has benn moved elsewhere. The path should be {src}."
    else:
        img = src

    crit_shorts_mask=np.load("../temp/crit_shorts_mask.npy")
    non_crit_shorts_mask=np.load("../temp/non_crit_shorts_mask.npy")
    crit_endpoints_mask=np.load("../temp/crit_endpoints_mask.npy")
    non_crit_endpoints_mask=np.load("../temp/non_crit_endpoints_mask.npy")
    pads_mask=np.load("../temp/pads_mask.npy")

    masks = [crit_shorts_mask,non_crit_shorts_mask,crit_endpoints_mask,non_crit_endpoints_mask,pads_mask]

    state = {
        'current_view': img.copy(),
        'offset_x': 0,
        'offset_y': 0,
        'current_mode': 0
    }
    # 0 pour vue normale
    # 1 pour courts-circuits critiques 
    # 2 pour courts-circuits non critiques
    # 3 pour points d'arrivée mal câblés, 
    # 4 pour tous points d'arrivée
    # 5 pour tous points d'arrivée + pads
    # 6 pour la vue avec tout

    if lang == "en" :
        cv.namedWindow("Main View", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Main View", mouse_callback, param=state)

        console.rule("[bold blue]MAGNIFYING GLASS")
        console.print("")
        console.print("[blue]Hover your mouse over the main view to display the zoomed-in version.")
        console.print("[blue][bold]Drag and drop[/bold] to zoom.")
        console.print("[blue]Press [bold]'r'[/bold] to reset the view.") 
        console.print("[blue]Press [bold]'s'[/bold] to save the view.")
        console.print("[blue]Press [bold]'q'[/bold] to quit.")
        console.print("[red]***")
        console.print("[blue]Press [bold]'1'[/bold] for the [bold white]unmodified[/bold white] view.")
        console.print("[blue]Press [bold]'2'[/bold] for the view showing the [bold red]critical short circuits[/bold red].")
        console.print("[blue]Press [bold]'3'[/bold] for the view showing the [bold dark_orange]non-critical short circuits[/bold dark_orange].")
        console.print("[blue]Press [bold]'4'[/bold] for the view showing the [bold dark_blue]incorrectly wired wire ends[/bold dark_blue].")
        console.print("[blue]Press [bold]'5'[/bold] for the wiew showing the [bold green]ends of all the wires[/bold green].")
        console.print("[blue]Press [bold]'6'[/bold] for the view showing the [bold cyan]tracks[/bold cyan].")
    
        console.print("\n[blue]Press [bold]'0'[/bold] for the view showing [bold magenta]all the changes[/bold magenta].\n")
    else :
        cv.namedWindow("Vue Principale", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Vue Principale", mouse_callback, param=state)

        console.rule("[bold blue]LOUPE")
        console.print("")
        console.print("[blue]Passer la souris sur la vue principale pour afficher la version zoomée.")
        console.print("[blue][bold]Glisser-déposer[/bold] pour zoomer.")
        console.print("[blue]Appuyer sur [bold]'r'[/bold] pour réinitialiser la vue.")
        console.print("[blue]Appuyer sur [bold]'s'[/bold] pour pour sauvegarder la vue.")
        console.print("[blue]Appuyer sur [bold]'q'[/bold] pour quitter.")
        console.print("[red]***")
        console.print("[blue]Appuyer sur [bold]'1'[/bold] pour la vue [bold white]non modifiée[/bold white].")
        console.print("[blue]Appuyer sur [bold]'2'[/bold] pour la vue avec les [bold red]courts-circuits critiques[/bold red].")
        console.print("[blue]Appuyer sur [bold]'3'[/bold] pour la vue avec les [bold dark_orange]courts-circuits non critiques[/bold dark_orange].")
        console.print("[blue]Appuyer sur [bold]'4'[/bold] pour la vue avec les [bold dark_blue]terminaisons des fils mal câblés[/bold dark_blue].")
        console.print("[blue]Appuyer sur [bold]'5'[/bold] pour la vue avec les [bold green]terminaisons de tous les fils[/bold green].")
        console.print("[blue]Appuyer sur [bold]'6'[/bold] pour la vue avec les [bold cyan]pistes[/bold cyan].")
    
        console.print("\n[blue]Appuyer sur [bold]'0'[/bold] pour la vue avec [bold magenta]toutes les modifications[/bold magenta].\n")

    while True:
        if lang == "en" :
            cv.imshow("Main View", state['current_view'])
        else :
            cv.imshow("Vue Principale", state['current_view'])
        key = cv.waitKey(1) & 0xFF 
        
        if key == ord('q'): # Quitter
            break
        elif key == ord('r'): # Reset 

            if state['current_mode'] == 0:
                state['current_view'] = img.copy()
            elif state['current_mode'] == 1:
                state['current_view'] = create_masked_view(img,crit_shorts_mask,colors[0])
            elif state['current_mode'] == 2:
                state['current_view'] = create_masked_view(img,non_crit_shorts_mask,colors[1])
            elif state['current_mode'] == 3:
                state['current_view'] = create_masked_view(img,crit_endpoints_mask,colors[2])
            elif state['current_mode'] == 4:
                state['current_view'] = create_masked_view(img,non_crit_endpoints_mask,colors[3])
            elif state['current_mode'] == 5:
                state['current_view'] = create_masked_view(img,pads_mask,colors[4])
            elif state['current_mode'] == 6:
                state['current_view'] = create_view_all_masks(img,masks,colors)

            state['offset_x'], state['offset_y'] = 0, 0

        elif key == ord('s'):
            t = strftime("%d-%m-%Y-%H-%M-%S", gmtime())
            outpath = "../output/output" + str(t) + ".png"
            cv.imwrite(outpath, state['current_view'])
            if lang == 'en' : console.print(f"[green]Image successfully saved to {outpath}")
            else : console.print(f"[green]Image sauvegardée avec succès à {outpath}")

        elif key == ord('&') or key == ord('1'): # Normal view
            state['current_mode'] = 0
            h,w = state['current_view'].shape[:2]
            state['current_view'] = img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w]
        elif key == ord('é') or key== ord('2'): # courts-circuits critiques
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_masked_view(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       crit_shorts_mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       colors[0])
            state['current_mode'] = 1
        elif key == ord('"') or key== ord('3'): # courts-circuits non critiques
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_masked_view(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       non_crit_shorts_mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       colors[1])
            state['current_mode'] = 2
        elif key == ord("'") or key== ord('4'): # points d'arrivée mal câblés
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_masked_view(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       crit_endpoints_mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       colors[2])
            state['current_mode'] = 3
        elif key == ord('(') or key== ord('5'): # tous les points d'arrivée bien câblés
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_masked_view(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       non_crit_endpoints_mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       colors[3])
            state['current_mode'] = 4
        elif key == ord('-') or key== ord('6'): # toutes les pistes
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_masked_view(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       pads_mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       colors[4])
            state['current_mode'] = 5
        elif key == ord('à') or key== ord('0'): # tout
            h,w = state['current_view'].shape[:2]
            state['current_view'] = create_view_all_masks(img[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w], 
                                                       [mask[state['offset_y']:state['offset_y']+h, state['offset_x']:state['offset_x']+w] for mask in masks], 
                                                       colors)
            state['current_mode'] = 6
        

    cv.destroyAllWindows()