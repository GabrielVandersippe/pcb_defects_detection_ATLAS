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
pad1 = [[220,1000],[230,1130]]
pad2 = [[246,1000],[256,1130]]
pad3 = [[270,1000],[280,1130]]
pad4 = [[296,1051],[308,1130]]
pad5 = [[322,1066],[334,1130]]
pad6 = [[323,1037],[359,1052]]
pad7 = [[402,1050],[453,1130]]
pad8 = [[469,1050],[551,1130]]
pad9 = [[566,1050],[615,1130]]
pad10 = [[749,1036],[785,1052]]
pad11 = [[773,1063],[786,1130]]
pad12 = [[799,1050],[813,1130]]
pad13 = [[825,1050],[839,1130]]
pad14 = [[851,1050],[863,1130]]
pad15 = [[877,1050],[890,1130]]
pad16 = [[903,1050],[915,1130]]
pad17 = [[929,1065],[941,1130]]
pad18 = [[932,1037],[967,1051]]
pad19 = [[1009,1050],[1060,1130]]
pad20 = [[1075,1050],[1157,1130]]
pad21 = [[1171,1050],[1220,1130]]
pad22 = [[1132,1078],[1135,1130]]
pad23 = [[1354,1078],[1369,1130]]
pad24 = [[1432,1079],[1447,1131]]
pad25 = [[1464,1079],[1479,1131]]
pad26 = [[1496,1079],[1511,1131]]
pad27 = [[1529,1079],[1544,1131]]
pad28 = [[1561,1079],[1576,1131]]
pad29 = [[1593,1079],[1608,1131]]
pad30 = [[1758,1051],[1811,1131]]
pad31 = [[1824,1051],[1907,1131]]
pad32 = [[1920,1052],[1972,1132]]
pad33 = [[2014,1038],[2052,1052]]
pad34 = [[2041,1065],[2053,1132]]
pad35 = [[2065,1002],[2079,1132]]
pad36 = [[2090,1002],[2104,1132]]
pad37 = [[2128,1002],[2256,1051]]
pad38 = [[2366,1052],[2417,1132]]
pad39 = [[2431,1052],[2513,1132]]
pad40 = [[2528,1052],[2579,1132]]
pad41 = [[2651,1053],[2663,1131]]
pad42 = [[297,1000],[798,1024]]
pad43 = [[923,1000],[1297,1024]]
pad44 = [[1369,1038],[1710,1062]]
pad45 = [[1369,999],[2051,1025]]
pad46 = [[2284,1001],[2662,1025]]

pads = np.array([pad1,pad2,pad3,pad4,pad5,pad6,pad7,pad8,pad9,pad10,pad11,pad12,pad13,pad14,pad15,pad16,pad17,pad18,pad19,pad20,pad21,pad22,pad23,pad24,pad25,pad26,pad27,pad28,pad29,pad30,pad31,pad32,pad33,pad34,pad35,pad36,pad37,pad38,pad39,pad40,pad41,pad42,pad43,pad44,pad45,pad46])

pads_nouveau_repere = np.array([np.dot(mat_passage,pads[i]-np.flip(np.array([centre,centre]),axis=1)) for i in range(46)]).astype(np.int16)
