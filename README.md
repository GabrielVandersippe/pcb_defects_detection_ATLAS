Ceci est un logiciel en lignes de commandes pour vérifier le câblage des modules ITkPix.

Installations requises :

- opencv-python
- numpy
- matplotlib
- scipy
- rich

Utilisation :

- mettre l'image du pcb câblé dans le dossier "ModulePictures" (au format .jpg)
- mettre l'image du pcb avant câblage dans le même dossier (également au format .jpg)
- exécuter dans un terminal la commande suivante :
  python main.py check --input nom_du_fichier
