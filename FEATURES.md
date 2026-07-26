# Paramètres de Configuration de l'Agressivité

Ce document détaille les paramètres définis dans le fichier `aggressiveness_config.json`. Ces paramètres dictent "l'agressivité" des algorithmes de vision par ordinateur utilisés pour isoler les fils et détecter les courts-circuits. L'ajustement de ces valeurs modifie la sensibilité des opérations de masquage, de seuillage et de morphologie.

## Paramètres de `wire_threshold`

Cette fonction est responsable de l'extraction de la forme des fils de l'arrière-plan en utilisant un filtrage des couleurs HSV, un seuillage adaptatif en niveaux de gris et un nettoyage morphologique.

### Masque de Saturation (`satmask_thr`)
Filtre les pixels en fonction de l'intensité de leur couleur (saturation dans l'espace colorimétrique HSV) pour supprimer les éléments d'arrière-plan très colorés (comme les zones jaunes ou gris foncé).
*   **`lb` (Limite Inférieure) :** Valeur de saturation minimale (ex. `0`).
*   **`ub` (Limite Supérieure) :** Valeur de saturation maximale (ex. `40`). Diminuer cette valeur rend le filtre plus agressif, supprimant tout ce qui a ne serait-ce qu'une légère teinte.

### Masque de Valeur (`valmask_thr`)
Filtre les pixels en fonction de leur luminosité (Valeur dans l'espace colorimétrique HSV) pour s'assurer que seuls les segments de fil très réfléchissants/lumineux sont conservés.
*   **`lb` (Limite Inférieure) :** Valeur de luminosité minimale (ex. `190`). Augmenter cette valeur exige que le fil soit strictement plus lumineux pour être détecté.
*   **`ub` (Limite Supérieure) :** Valeur de luminosité maximale (ex. `255`). 

### Masque d'Intensité Adaptatif (`intensitymask`)
Applique un seuillage adaptatif sur une version floutée en niveaux de gris de l'image pour tenir compte de l'éclairage inégal sur l'image.
*   **`blocksize` (Taille de bloc) :** La taille du voisinage de pixels utilisé pour calculer le seuil d'un pixel spécifique (ex. `25`). Doit être un nombre impair. Une taille de bloc plus petite rend le seuillage plus localisé et plus sensible aux petites variations de lumière.
*   **`cst` (Constante) :** Une constante soustraite de la moyenne du voisinage (ex. `-10`). Une valeur négative abaisse effectivement le seuil, rendant l'algorithme plus permissif (moins agressif) dans la conservation des pixels.

### Noyaux Morphologiques
Utilisés pour nettoyer le masque binaire, supprimer le bruit ("blobs") et combler les petits espaces dans les fils détectés.
*   **`closing_kernel` :** Un tableau `[largeur, hauteur]` (ex. `[5, 1]`) définissant un élément structurant rectangulaire. Il est utilisé dans une opération de *dilatation* (fermeture) pour combler horizontalement les petits espaces dans les fils détectés.
*   **`kernel_wheel_shape` :** Un tableau `[largeur, hauteur]` (ex. `[35, 9]`) utilisé pour générer dynamiquement une série de noyaux personnalisés et pivotés (`kernel_wheel`). Ceux-ci sont appliqués dans une opération d'*ouverture* morphologique pour éliminer de manière agressive les blobs irréguliers tout en préservant les formes linéaires des fils.

---

## Paramètres de `find_shorts`

Cette fonction prend le masque binaire nettoyé et utilise les composantes connexes ainsi que des opérations morphologiques pour trouver les courts-circuits physiques entre les fils adjacents.

### Connectivité
*   **`connectivity` :** Définit comment les pixels sont regroupés en composantes connexes (généralement `4` ou `8`). 
    *   Une valeur de `4` signifie que les pixels ne sont connectés que s'ils partagent des bords horizontaux ou verticaux. 
    *   Une valeur de `8` est moins agressive, permettant aux connexions diagonales de former une seule composante.

### Noyaux Morphologiques pour la Détection des Courts-Circuits
*Remarque : Bien que définis dans la configuration JSON, vérifiez que votre script Python passe ces `kwargs` au lieu de coder en dur les appels `cv.getStructuringElement` dans cette fonction.*
*   **`vkernel` :** Un tableau `[largeur, hauteur]` (ex. `[1, 7]`) définissant un noyau rectangulaire vertical. Il est utilisé pour une opération d'*érosion*. Étant donné que les fils standards sont fins horizontalement, ce grand noyau vertical érode complètement les fils normaux de manière agressive, ne laissant derrière lui que les zones plus épaisses et plus larges où se produisent les courts-circuits.
*   **`hkernel` :** Un tableau `[largeur, hauteur]` (ex. `[25, 3]`) définissant un noyau rectangulaire horizontal. Il est utilisé pour une opération de *dilatation* appliquée aux zones de court-circuit restantes. Cela relie de manière agressive les points de contact proches afin de minimiser le nombre de cercles de détection en double ou se chevauchant pour un seul court-circuit.
