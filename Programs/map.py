import numpy as np

# trim_nb = (0, 0, 0, 0)  # trim number between 0 and 15 (used only once in bounding_map_trim line 175)

map1 = np.zeros(198, dtype=int) # map[i-1] is the right position for pad i of GA1
map2 = np.zeros(198, dtype=int) # map[i-1] is the right position for pad i of GA2
map3 = np.zeros(198, dtype=int) # map[i-1] is the right position for pad i of GA3
map4 = np.zeros(198, dtype=int) # map[i-1] is the right position for pad i of GA4

def trim (trim_, map_) :
    trim_list = np.array([[1, 1, 1, 1], 
                 [0, 1, 1, 1], 
                 [1, 0, 1, 1], 
                 [0, 0, 1, 1], 
                 [1, 1, 0, 1], 
                 [0, 1, 0, 1], 
                 [1, 0, 0, 1], 
                 [0, 0, 0, 1],
                 [1, 1, 1, 0],
                 [0, 1, 1, 0],
                 [1, 0, 1, 0],
                 [0, 0, 1, 0],
                 [1, 1, 0, 0],
                 [0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0]]) # trim_list[i] is the bounding of trim pads for trim i (1 if connected, 0 if not)
    map_[46:50] = trim_list[trim_] * 7

def GA1_without_trim (map_ = map1) :
    for i in range (6) :
        map_[i] = i + 1
    map_[7:13] = 7
    map_[13:18] = 8
    map_[18:28] = 9
    map_[28:33] = 10
    map_[33:41] = 7
    map_[42:44] = 7
    for i in range (50,59) :
        map_[i] = 11 + i - 50
    map_[60:65] = 20
    map_[65:70] = 21
    map_[70:80] = 22
    map_[80:85] = 23
    map_[85:91] = 20
    map_[92:107] = [24, 25, 26, 27, 26, 27, 28, 29, 27, 30, 31, 27, 32, 33, 27]
    map_[109] = 27
    map_[110] = 26
    map_[112:117] = 27
    map_[117:122] = 34
    map_[122:132] = 35
    map_[132:137] = 36
    map_[137:142] = 27
    for i in range (143, 147) :
        map_[i] = 37 + i - 143
    map_[152:162] = 41 # Pulltest wires
    map_[165:170] = 42
    map_[170:175] = 43
    map_[175:185] = 44
    map_[185:190] = 45
    map_[190:197] = 42
    map_[197] = 46

def GA2_without_trim (map_ = map2) :
    for i in range (6) :
        map_[i] = i + 1
    map_[7:13] = 7
    map_[13:18] = 8
    map_[18:28] = 9
    map_[28:33] = 10
    map_[33:41] = 7
    map_[43] = 7 # diff GA1
    for i in range (50,59) :
        map_[i] = 11 + i - 50
    map_[60:65] = 20
    map_[65:70] = 21
    map_[70:80] = 22
    map_[80:85] = 23
    map_[85:91] = 20
    map_[92:101] = [24, 25, 26, 27, 26, 27, 28, 29, 27] # diff GA1
    map_[103:107] = [27, 30, 31, 27] # diff GA1
    map_[109] = 27
    map_[110] = 26
    map_[112:117] = 27
    map_[117:122] = 32 # diff GA1 (offset -2)
    map_[122:132] = 33 # diff GA1 (offset -2)
    map_[132:137] = 34 # diff GA1 (offset -2)
    map_[137:142] = 27
    for i in range (143, 147) :
        map_[i] = 35 + i - 143 # diff GA1 (offset -2)
    for i in range (148, 152) :
        map_[i] = 39 + i - 148 # diff GA1
    map_[155:160] = 43 # Pulltest wires # diff GA1
    map_[165:170] = 44 # diff GA1 (offset +2)
    map_[170:175] = 45 # diff GA1 (offset +2)
    map_[175:185] = 46 # diff GA1 (offset +2)
    map_[185:190] = 47 # diff GA1 (offset +2)
    map_[190:197] = 44 # diff GA1 (offset +2)
    map_[197] = 48 # diff GA1 (offset +2)

def GA3_without_trim (map_ = map3) :
    for i in range (6) :
        map_[i] = i + 1
    map_[7:13] = 7
    map_[13:18] = 8
    map_[18:28] = 9
    map_[28:33] = 10
    map_[33:41] = 7
    map_[42] = 7 # diff GA1
    for i in range (50,59) :
        map_[i] = 11 + i - 50
    map_[60:65] = 20
    map_[65:70] = 21
    map_[70:80] = 22
    map_[80:85] = 23
    map_[85:91] = 20
    map_[92:111] = [24, 25, 26, 27, 26, 27, 28, 29, 27, 30, 31, 27, 32, 33, 27, 34, 35, 27, 26] # diff GA1
    map_[112:117] = 27
    map_[117:122] = 36 # diff GA1 (offset +2)
    map_[122:132] = 37 # diff GA1 (offset +2)
    map_[132:137] = 38 # diff GA1 (offset +2)
    map_[137:142] = 27
    for i in range (143, 147) :
        map_[i] = 39 + i - 143 # diff GA1 (offset +2)
    map_[150:160] = 43 # Pulltest wires # diff GA1 (offset +2)
    map_[165:170] = 44 # diff GA1 (offset +2)
    map_[170:175] = 45 # diff GA1 (offset +2)
    map_[175:185] = 46 # diff GA1 (offset +2)
    map_[185:190] = 47 # diff GA1 (offset +2)
    map_[190:197] = 44 # diff GA1 (offset +2)
    map_[197] = 48 # diff GA1 (offset +2)

def GA4_without_trim (map_ = map4) :
    for i in range (6) :
        map_[i] = i + 1
    map_[7:13] = 7
    map_[13:18] = 8
    map_[18:28] = 9
    map_[28:33] = 10
    map_[33:41] = 7
    # diff GA1
    for i in range (50,59) :
        map_[i] = 11 + i - 50
    map_[60:65] = 20
    map_[65:70] = 21
    map_[70:80] = 22
    map_[80:85] = 23
    map_[85:91] = 20
    map_[92:101] = [24, 25, 26, 27, 26, 27, 28, 29, 27] # diff GA1
    map_[103] = 27 # diff GA1
    map_[106] = 27 # diff GA1
    map_[109] = 27
    map_[110] = 26
    map_[112:117] = 27
    map_[117:122] = 30 # diff GA1 (offset -4)
    map_[122:132] = 31 # diff GA1 (offset -4)
    map_[132:137] = 32 # diff GA1 (offset -4)
    map_[137:142] = 27
    for i in range (143, 147) :
        map_[i] = 33 + i - 143 # diff GA1 (offset -4)
    for i in range (148, 156) :
        map_[i] = 37 + i - 148 # diff GA1
    map_[165:170] = 45 # diff GA1 (offset +3)
    map_[170:175] = 46 # diff GA1 (offset +3)
    map_[175:185] = 47 # diff GA1 (offset +3)
    map_[185:190] = 48 # diff GA1 (offset +3)
    map_[190:197] = 45 # diff GA1 (offset +3)
    map_[197] = 49 # diff GA1 (offset +3)

def bounding_map_without_trim () :
    GA1_without_trim()
    GA2_without_trim()
    GA3_without_trim()
    GA4_without_trim()

def bounding_map_trim (trim_) : # trim est un tuple de 4 éléments (GA1, GA2, GA3, GA4)
    trim(trim_[0],map1)
    trim(trim_[1],map2)
    trim(trim_[2],map3)
    trim(trim_[3],map4)

# bounding_map_without_trim()
# bounding_map_trim(trim_nb)

def list_pads_pistes (map_, offset_) : # offset_ = 1 pour GA1 ; 2 pour GA2 ; 3 pour GA3 et 4 pour GA4
    result = []
    for i in range (len(map_)) :
        x = map_[i]
        if x != 0 :
            result.append((i+1 + offset_*1000, int(x) + offset_*100))
    return (result)

def bounding_map_pads_pistes () :
    result = list_pads_pistes(map1, 1)
    result = result + list_pads_pistes(map2, 2)
    result = result + list_pads_pistes(map3, 3)
    result = result + list_pads_pistes(map4, 4)
    return (result)


# pads_pistes = bounding_map_pads_pistes() # liste des connections attendues pour les fils