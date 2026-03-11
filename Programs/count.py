import cv2
import scipy.signal

# Theoretical number of wires

def extract_serial_number (file_name) :
    """Extracts the serial number (starting with 20UPGM) of a module

    Arguments :

    file_name - string : name of the file with the serial number in it

    Returns : string
    """
    names = file_name.split("_")
    for x in names :
        if "20UPGM" in x :
            if "/" in x :
                return(x.split("/")[-1])
            else :
                return (x)
    assert False, "No serial number found in the file name"


def iref_trim (serial_number, data) :
    """Reads the iref of a module

    Arguments :

    serial_number - string : serial number of the module

    data - list[dict] : iref of each module

    Returns : (int, int, int, int)
    """
    ok = False
    for x in data :
        if x['serialNumber'] == serial_number :
            iref = x
            ok = True
    assert ok, "Serial number not found"
    return (iref['IREF_TRIM_1'], iref['IREF_TRIM_2'], iref['IREF_TRIM_3'], iref['IREF_TRIM_4'])


def expected_wire_number (serial_number, data) :
    """Calculates the theoretical number of wires of a module

    Arguments :

    serial_number - string : serial number of the module

    data - list[dict] : iref of each module

    Returns : int
    """
    iref = iref_trim(serial_number, data)
    nb_wire_per_trim = [4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0] # number of wires expected depending on iref
    return (693 + nb_wire_per_trim[iref[0]] + nb_wire_per_trim[iref[1]] + nb_wire_per_trim[iref[2]] + nb_wire_per_trim[iref[3]]) # 693 wires independant of iref + number of iref wires


# Counting the real number of wires

def crop_ligns (image, level = 0.55) :
    """Determines the limits of the vertical zone of interest of the picture

    Arguments :

    image - array of pixels : the working image

    level (optional) - int : limit of greenness before the algorithm stops

    Returns : (int, int)
    """
    green = image[:,:,1].sum(axis=1) / (image[:,:,0].sum(axis=1) + image[:,:,2].sum(axis=1))
    n = image.shape[0]

    limit_high = 0
    x = green[limit_high]
    while (limit_high < n//2) and (x <= level) :
        limit_high+=1
        x = green[limit_high]
    
    limit_low = n-1
    x = green[limit_low]
    while (limit_low > n//2) and (x <= level) :
        limit_low-=1
        x = green[limit_low]
    
    return (limit_high + 25, limit_low - 22) # +25 and -22 : gaps between the green area and the wire zone


def count (image_grey_crop, column) :
    """Count the number of wires in a specific column

    Arguments :

    image_grey_crop - array of pixels : the working greyscale image, cropped vertically

    column - int : column of interest

    Returns : int
    """
    peaks, _ = scipy.signal.find_peaks(image_grey_crop[:,column], distance=3, prominence=50, height=190, width=(0,9)) # appropriate parameters were determined with already existing datasets
    return (peaks.shape[0])


def crop_columns_left (image_grey_crop, level = 100) :
    """Determines the left limit of the wire zone

    Arguments :

    image_grey_crop - array of pixels : the working greyscale image, cropped vertically

    level (optionnal) - int : number of wires required before the algorithm stops

    Returns : int
    """
    n = image_grey_crop.shape[1]
    left = 0
    count_left = count(image_grey_crop, left)
    while (count_left < level) and (left < n) :
        left += 1
        count_left = count(image_grey_crop, left)
    return (left)


def crop_columns_right (image_grey_crop, level = 100) :
    """Determines the right limit of the wire zone

    Arguments :

    image_grey_crop - array of pixels : the working greyscale image, cropped vertically

    level (optionnal) - int : number of wires required before the algorithm stops

    Returns : int
    """
    n = image_grey_crop.shape[1]
    right = n - 1
    count_right = count(image_grey_crop, right)
    while (count_right < level) and (right > 0) :
        right -= 1
        count_right = count(image_grey_crop, right)
    return (right)


# Final fonctions

def test_wire_number (file_name, data) :
    """Tests whether the number of wires on a module is correct or not

    Arguments :

    file_name - string : name of the file (the image must be in the same folder as this programm)

    data - list[dict] : iref of each module

    Returns : 
    test - bool : True if the number of wires is correct, False otherwise
    expected_nb - int : theoretical number of wires
    real_nb - int : number of wires counted
    """
    expected_nb = expected_wire_number(extract_serial_number(file_name), data)

    image = cv2.imread(file_name)
    n = image.shape[1]

    high_left, low_left = crop_ligns(image[:,:n//2])
    high_right, low_right = crop_ligns(image[:,n//2:])
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left = crop_columns_left(grey[high_left:low_left])
    right = crop_columns_right(grey[high_right:low_right])

    real_nb_left = count(grey[high_left:low_left], column = left+35) # +35 : gap between the left limit of the wire zone and the column with every wires
    real_nb_right = count(grey[high_right:low_right], column = right-35) # -35 : gap between the right limit of the wire zone and the column with every wires
    real_nb = real_nb_left + real_nb_right
    test = expected_nb == real_nb

    return (test, expected_nb, real_nb)


def wire_pos (image) :
    """Gives the position of every wire detected

    Arguments :

    image - array of pixels : the working image

    Returns :
    
    real_peaks_left - array of int : the ordinates of the wires on the left side

    real_left - int : the abscissa of the wires on the left side (same for all wires)

    real_peaks_right - array of int : the ordinates of the wires on the right side

    real_right - int : the abscissa of the wires on the right side (same for all wires)
    """
    n = image.shape[1]

    high_left, low_left = crop_ligns(image[:,:n//2])
    high_right, low_right = crop_ligns(image[:,n//2:])
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left = crop_columns_left(grey[high_left:low_left])
    right = crop_columns_right(grey[high_right:low_right])

    peaks_left, _ = scipy.signal.find_peaks(grey[high_left:low_left, left+35], distance=3, prominence=50, height=190, width=(0,9)) # +35 : gap between the left limit of the wire zone and the column with every wires
    peaks_right, _ = scipy.signal.find_peaks(grey[high_right:low_right, right-35], distance=3, prominence=50, height=190, width=(0,9)) # -35 : gap between the right limit of the wire zone and the column with every wires
    real_peaks_left, real_left, real_peaks_right, real_right = peaks_left + high_left, left+35, peaks_right + high_right, right-35 # returning to the coordinates on the original picture

    return(real_peaks_left, real_left, real_peaks_right, real_right)

def test_wire_finding (file_name) :
    """Tests the wire finding algorithm by creating a test image with the detected wires marked by a black pixel

    Arguments :

    file_name - string : name of the file (the image must be in the same folder as this programm)

    Returns : None
    """
    image = cv2.imread(file_name)
    image2 = image.copy()
    test_left, left, test_right, right = wire_pos(image)
    for x in test_left :
        image2[x, left] = [0, 0, 0]
    for x in test_right :
        image2[x, right] = [0, 0, 0]
    cv2.imwrite('test.jpg',image2)