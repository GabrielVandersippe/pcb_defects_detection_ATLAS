Ceci est un logiciel en lignes de commandes pour vérifier le câblage des modules ITkPix.

# Prérequis :

- Installer `poetry` (https://python-poetry.org/)
- Entrer `poetry install` dans le dossier contenant le package. Ceci install les dépendances spécifiées par le fichier `poetry.lock`
- Sous Bash, entrer `eval $(poetry env activate)` pour entrer dans l'environnement nouvellement défini. Dans d'autres types de terminaux, se référer à la documentation officielle de poetry (https://python-poetry.org/docs/managing-environments/). Il se peut qu'il faille entrer `source chemin/vers/python` à la place pour l'activer.
- Pour s'assurer d'être dans le bon environnement, entrer `which python`. Si ce qui est retourné est le chemin spécifié, alors vous êtes bien dans l'environnement virtuel.
- Vous êtes prêt à exécuter le programme !

# Pour faire fonctionner le programme :

- Mettre l'image du module câblé _ET_ non câblé dans un même dossier (par défaut, dans `./ModulePictures`.)
- Se placer dans le dossier qui contient le fichier main.py (`./pcb_defect_detector/`)
- Définir le dossier dans lequel vous avez placé vos images (si modifié), via la commande `python main.py config --folder votre/chemin`
- Exécuter dans un terminal la commande suivante :
  `python main.py check --input nom_du_fichier`
- Les autres commandes peuvent être affichées via l'aide depuis la CLI.
- L'output sera enregistré dans `./output/`
