Ceci est un logiciel en lignes de commandes pour vérifier le câblage des modules ITkPix.

Installations requises :
- python : 3.13.2
- opencv-python : 4.13.0.92
- numpy : 2.4.4
- matplotlib : 3.10.8
- scipy : 1.17.1

Utilisation :
- mettre l'image du pcb câblé dans le dossier "ModulePictures" (au format .jpg)
- mettre l'image du pcb avant câblage dans le même dossier (également au format .jpg)
- exécuter dans un terminal la commande suivante :
    python main.py check --input nom_du_fichier

Le programme accepte également la référence du module ou le chemin complet vers l'image du pcb câblé.
Exemples :
    python main.py check --input P1004_20UPGM23211223
    python main.py check --input P1004_20UPGM23211223_AfterBonding_NOK
    python main.py check --input P1004_20UPGM23211223_AfterBonding_NOK.jpg
    python main.py check --input ModulePictures/P1004_20UPGM23211223_AfterBonding_NOK.jpg

Il est également possible d'ajouter un iref (par exemple si celui-ci n'est pas précisé dans Reference/iref_trim_per_module_v2.json).
Exemple :
    python main.py check --input P1004_20UPGM23211223 --iref 10,8,9,9

Vous pouvez également modifier le fichier Configuration/config.json pour changer :
- pictures_folder : le dossier dans lequel déposer les images (ModulePictures par défaut)
- pictures_format : le format des images (.jpg par défaut)
- suffix_after_bonding : le suffixe des images après câblage (AfterBonding_NOK par défaut)
- suffix_before_bonding : le suffixe des images avant câblage (Reception_Glo_NoLight_AfterClean par défaut)
- language : la langue de l'interface ("en" pour anglais, "fr" pour français)
- zoom : la puissance du zoom de la loupe (4 par défaut)