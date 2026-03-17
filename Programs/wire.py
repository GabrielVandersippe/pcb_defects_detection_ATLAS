import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import json
from time import time

from count import expected_wire_number, extract_serial_number, wire_pos

# open the json with the iref for each module

# with open("ModulePictures/iref_trim_per_module_v2.json", "r") as f:
#     data = json.load(f)

# Utils functions for analyzing pixels in general

def norm(pix: np.ndarray) -> float:
    """Computes the norm of an rgb pixel.

    Arguments :
    pix - array of floats : the pixel.

    Returns : float
    """
    return float(pix[0])**2 + float(pix[1])**2 + float(pix[2])**2


def isWhite(img: np.ndarray, coord: tuple, threshold = 100000) -> bool:
    """Tests whether or not a pixel is considered white, using an arbitrary threshold.

    Arguments :

    img - array of pixels : the working image

    coord - tuple of ints : the coordinates of the pixel

    threshold - float : arbitrary threshold

    Returns : bool
    """
    return norm(img[coord[0],coord[1],:]) > threshold


def validCoord(img: np.ndarray, coord: tuple) -> bool:
    """Tests whether or not the coordinates are within the boundaries of the image.

    Arguments :

    img - array of pixels : the working image

    coord - tuple of ints : the coordinates of the working pixel

    Returns : bool
    """
    return 0 <= coord[0] and coord[0] < len(img) and 0 <= coord[1] and coord[1] < len(img[0])



# Intermediate functions for the Breadth-First Search (BFS) wire recognition function

def neighboursList(img: np.ndarray, coord: tuple) -> list:
    """Creates the list of coordinates of the neighbouring white pixels given the current pixel.

    Arguments :

    img - array of pixels : the working image

    coord - tuple of ints : the coordinates of the current pixel

    Returns : list of coordinates
    """
    directions_list = [(1,0),(-1,0),(0,1),(0,-1)]
    neighbours_list = []
    for direction in directions_list:
        neighbour_coord = (coord[0] + direction[0], coord[1] + direction[1])
        if validCoord(img,neighbour_coord) and isWhite(img,neighbour_coord):
            neighbours_list.append(neighbour_coord)
    return neighbours_list


def visit(img: np.ndarray, coord: tuple, wire: list,queue: deque):
    """Visits (i.e. adds to the visiting queue) all the unvisited neighbouring white pixels given the current pixel.

    Arguments :

    img - array of pixels : the working image

    coord - tuple of ints : the coordinates of the current pixel

    wire - list of coordinates : the list of pixels already visited

    queue - deque of coordinates : the visiting queue
    """
    neighbours_list = neighboursList(img,coord)
    for neighbour in neighbours_list:
        if (neighbour not in wire) and (neighbour not in queue):
            queue.append(neighbour)



# Breadth-First Search (BFS) wire recognition function

def bfsWire(img: np.ndarray, start_coord: tuple) -> list:
    """Creates the list of coordinates of all the pixels of a wire given a starting pixel, using a Breadth-First Search (BFS) algorithm.

    Arguments : 

    img - array of pixels : the working image

    start_coord - tuple of ints : the coordinates of the starting pixel

    Returns : list of coordinates
    """
    wire = []
    queue = deque()
    queue.append(start_coord)
    while len(queue) != 0:
        current_coord = queue[0]
        if current_coord not in wire:
            wire.append(current_coord)
        visit(img,current_coord,wire,queue)
        queue.popleft()
    return wire



# Finding the edges coordinates of a given wire

def extremeCoords(wire: list, side = "left") -> int:
    """Finds the index of either the leftmost or the rightmost pixel of a wire, depending on the input side.

    Arguments :

    wire -- list of coordinates : the list of all pixels in the wire

    side -- string : the side chosen for finding the extreme coordinate, by default : left

    Returns : int
    """
    directions = {"left":-1, "right":1}
    index = 0
    for i in range(1,len(wire)):
        if directions[side] * (wire[i][1] - wire[index][1]) > 0:
            index = i
    return index


def wireEdges(wire: list,) -> tuple:
    """Gives the coordinates of both the leftmost and the rightmost pixels of a wire.

    Arguments :

    wire -- list of coordinates : the list of all pixels in the wire

    Returns : tuple of coordinates
    """
    i_left , i_right = extremeCoords(wire, "left"), extremeCoords(wire, "right")
    return (wire[i_left], wire[i_right])



# Checking if a wire is touching another

def isTouching(wire: list, threshold = 2900) -> bool:
    """Tests whether or not a wire is touching another by checking if it contains too many pixels.

    Arguments :

    wire -- list of coordinates : the list of all pixels in the wire

    threshold - int : arbitrary threshold for how many pixels is too many pixels

    Returns : bool
    """
    return len(wire) > threshold


# Combining start pixel detection, wire counting and wire plotting


def analyseWires(filename: str):
    """Highlights all the wires in an image, with different colors if a pixel is touching another.

    Arguments : 

    filename - str : the file name of the working image
    """
    count = 0
    t = time()
    img = plt.imread(filename)
    n_expected = expected_wire_number(extract_serial_number(filename),data)
    copy = img.copy()
    (x_list_left,y_left,x_list_right,y_right) = wire_pos(img)
    n_detected = len(x_list_left) + len(x_list_right)
    for (x_list, y) in [(x_list_left,y_left),(x_list_right,y_right)]:
        for x in x_list:
            if count%10 == 0 :
                print("Status : " + str(int(100 * count/n_detected)) + " %")
                print("Time spent : " + str(int(time() - t)) + "seconds")
            count+=1
            wire = bfsWire(img,(x,y))
            if isTouching(wire):
                for pixel in wire:
                    copy[pixel[0],pixel[1],:] = np.array([0,0,255])
            edges = wireEdges(wire)
    cv2.imwrite("result.jpg",copy)
    print("Wires expected : " + str(n_expected))
    print("Wires detected : " + str(n_detected))

#analyseWires("ModulePictures\P1004_20UPGM23211223_AfterBonding_NOK.jpg")