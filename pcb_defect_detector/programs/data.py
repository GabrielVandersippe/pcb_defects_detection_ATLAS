import numpy as np

###TODO : utiliser un ficher JSON à la place 

# CONTIENT DES DONNES MESUREES A LA MAIN SUR L'IMAGE 'ModulePictures/20UPGM23210250_PPPV2_45_AfterBonding_NOK.jpg'

## DATA IMAGE 1

## POSITIONS MIRES :
mire1 = [251, 927]
mire2 = [5234, 932]
mire3 = [2703, 1058]
mire4 = [2796, 1058]
mire5 = [247, 5788]
mire6 = [5230, 5788]
mire7 = [2700, 5659]
mire8 = [2793, 5659]

mires_img1 = np.array([mire1, mire2, mire3, mire4, mire5, mire6, mire7, mire8])

centre = np.flip(np.mean(mires_img1, axis = 0).astype(np.int16)) #Pour avoir le centre en x,y

dilat_ref = 0.5*(np.linalg.norm(mires_img1[0]-mires_img1[5]) + np.linalg.norm(mires_img1[1]-mires_img1[4]))

a = sum([(mires_img1[i+4,0]-mires_img1[i,0])/(mires_img1[i+4,1]-mires_img1[i,1]) for i in range(0,4)])/4 #Pente horizontale moyenne

mat_passage = 1/np.sqrt(a**2 + 1) * np.array([[1,-a],[a,1]])

## PADS DU PCB (cf carte_noms_pads.pdf pour leur numéro)
pad101 = [[220,1000],[230,1130]]
pad102 = [[246,1000],[256,1130]]
pad103 = [[270,1000],[280,1130]]
pad104 = [[296,1051],[308,1130]]
pad105 = [[322,1066],[334,1130]]
pad106 = [[323,1037],[359,1052]]
pad108 = [[402,1050],[453,1130]]
pad109 = [[469,1050],[551,1130]]
pad110 = [[566,1050],[615,1130]]
pad111 = [[749,1036],[785,1052]]
pad112 = [[773,1063],[786,1130]]
pad113 = [[799,1050],[813,1130]]
pad114 = [[825,1050],[839,1130]]
pad115 = [[851,1050],[863,1130]]
pad116 = [[877,1050],[890,1130]]
pad117 = [[903,1050],[915,1130]]
pad118 = [[929,1065],[941,1130]]
pad119 = [[932,1037],[967,1051]]
pad121 = [[1009,1050],[1060,1130]]
pad122 = [[1075,1050],[1157,1130]]
pad123 = [[1171,1050],[1220,1130]]
pad124 = [[1132,1078],[1135,1130]]
pad125 = [[1354,1078],[1369,1130]]
pad128 = [[1432,1079],[1447,1131]]
pad129 = [[1464,1079],[1479,1131]]
pad130 = [[1496,1079],[1511,1131]]
pad131 = [[1529,1079],[1544,1131]]
pad132 = [[1561,1079],[1576,1131]]
pad133 = [[1593,1079],[1608,1131]]
pad134 = [[1758,1051],[1811,1131]]
pad135 = [[1824,1051],[1907,1131]]
pad136 = [[1920,1052],[1972,1132]]
pad137 = [[2014,1038],[2052,1052]]
pad138 = [[2041,1065],[2053,1132]]
pad139 = [[2065,1002],[2079,1132]]
pad140 = [[2090,1002],[2104,1132]]
pad141 = [[2128,1002],[2256,1051]]
pad143 = [[2366,1052],[2417,1132]]
pad144 = [[2431,1052],[2513,1132]]
pad145 = [[2528,1052],[2579,1132]]
pad146 = [[2651,1053],[2663,1131]]
pad107 = [[297,1000],[798,1024]]
pad120 = [[923,1000],[1297,1024]]
pad126 = [[1369,1038],[1710,1062]]
pad127 = [[1369,999],[2051,1025]]
pad142 = [[2284,1001],[2662,1025]]

pads = np.array([pad101,pad102,pad103,pad104,pad105,pad106,pad107,pad108,pad109,pad110,pad111,pad112,pad113,pad114,pad115,pad116,pad117,pad118,pad119,pad120,pad121,pad122,pad123,pad124,pad125,pad126,pad127,pad128,pad129,pad130,pad131,pad132,pad133,pad134,pad135,pad136,pad137,pad138,pad139,pad140,pad141,pad142,pad143,pad144,pad145,pad146])

pads_nouveau_repere = np.array([np.dot(mat_passage,pads[i]-np.flip(np.array([centre,centre]),axis=1)) for i in range(46)]).astype(np.int16)