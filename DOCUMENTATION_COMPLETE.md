# Cours complet — Détection de palmiers et maisons sur orthophotos avec YOLOv8

Ce document est un cours progressif qui vous accompagne de zéro jusqu'à l'utilisation concrète d'un modèle de détection d'objets dans un logiciel de cartographie. Il s'adresse à des personnes qui n'ont jamais fait de deep learning ni de vision par ordinateur. Chaque notion nouvelle est expliquée au fil du texte, en phrases complètes, avec des exemples concrets tirés du projet. L'objectif est que vous compreniez non seulement *quoi* faire à chaque étape, mais aussi *pourquoi* on le fait et *comment* les choses fonctionnent en coulisses.

Le projet qui sert de fil conducteur consiste à détecter automatiquement des palmiers et des maisons sur des images aériennes géoréférencées, puis à exploiter ces détections comme des couches vectorielles dans le logiciel QGIS. Tout le parcours — de la préparation des images jusqu'à la carte finale — est couvert ici.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Les bases : qu'est-ce que la détection d'objets ?](#2-les-bases--quest-ce-que-la-détection-dobjets-)
3. [L'apprentissage supervisé : comment un modèle apprend](#3-lapprentissage-supervisé--comment-un-modèle-apprend)
4. [YOLO : fonctionnement, historique et variantes](#4-yolo--fonctionnement-historique-et-variantes)
5. [Préparation des données : découpe des orthophotos en tuiles](#5-préparation-des-données--découpe-des-orthophotos-en-tuiles)
6. [Annotation des images avec LabelImg](#6-annotation-des-images-avec-labelimg)
7. [Organisation du dataset et fichier de configuration](#7-organisation-du-dataset-et-fichier-de-configuration)
8. [Scripts du projet : prepare_dataset et remap_labels](#8-scripts-du-projet--prepare_dataset-et-remap_labels)
9. [Le notebook d'entraînement sur Google Colab](#9-le-notebook-dentraînement-sur-google-colab)
10. [Paramètres d'entraînement expliqués un par un](#10-paramètres-dentraînement-expliqués-un-par-un)
11. [Comprendre les métriques : précision, rappel, mAP](#11-comprendre-les-métriques--précision-rappel-map)
12. [Export du modèle au format ONNX](#12-export-du-modèle-au-format-onnx)
13. [Ajout des métadonnées Deepness](#13-ajout-des-métadonnées-deepness)
14. [Utilisation dans QGIS avec le plugin Deepness](#14-utilisation-dans-qgis-avec-le-plugin-deepness)
15. [Résumé de la chaîne complète et conseils pratiques](#15-résumé-de-la-chaîne-complète-et-conseils-pratiques)
16. [Glossaire](#16-glossaire)

---

## 1. Vue d'ensemble du projet

### Le problème à résoudre

Imaginez que l'on vous donne une photographie aérienne couvrant plusieurs kilomètres carrés de territoire : des champs, des routes, des habitations et des palmiers éparpillés partout. Votre mission est de compter chaque palmier et chaque maison, et de les situer précisément sur une carte. À la main, ce travail prendrait des jours, voire des semaines, et serait rempli d'erreurs et d'oublis. C'est exactement le genre de tâche qu'un modèle de deep learning (apprentissage profond) peut automatiser.

Le **deep learning** est une branche de l'intelligence artificielle dans laquelle on entraîne un programme informatique — appelé **réseau de neurones** — à reconnaître des motifs dans des données (ici, des images). Une fois entraîné, ce programme peut analyser de nouvelles images qu'il n'a jamais vues et y repérer les objets qui l'intéressent en quelques minutes.

### Ce que le projet produit concrètement

Le résultat final n'est pas simplement une image annotée. On obtient des **couches vectorielles**, c'est-à-dire des fichiers contenant des polygones (des rectangles géoréférencés) que l'on peut ouvrir dans un logiciel de Système d'Information Géographique (**SIG**) comme QGIS. Chaque polygone représente un objet détecté (un palmier ou une maison) et porte des attributs : sa classe (« palmier » ou « maison ») et un score de confiance indiquant à quel point le modèle est sûr de sa détection. On peut alors filtrer, mesurer, colorier et superposer ces polygones à d'autres données géographiques.

### La chaîne de travail en un coup d'œil

Pour arriver à ce résultat, on suit une série d'étapes, toujours dans le même ordre :

- **Découpe** : les grandes orthophotos (images aériennes géoréférencées) sont découpées en petites tuiles de taille fixe, manipulables une par une.
- **Annotation** : pour chaque tuile, un humain dessine des rectangles autour des palmiers et des maisons à l'aide du logiciel LabelImg, et indique la classe de chaque objet.
- **Organisation du dataset** : les tuiles annotées sont réparties en deux groupes — un ensemble d'entraînement (80 %) et un ensemble de validation (20 %) — et rangées dans une arborescence de dossiers standardisée.
- **Entraînement** : un modèle YOLOv8 apprend à détecter les palmiers et les maisons en s'entraînant sur les images annotées. Cet entraînement se fait sur Google Colab, un service gratuit qui met à disposition un processeur graphique (GPU) puissant.
- **Export** : le modèle entraîné est converti au format ONNX, un format universel lisible par de nombreux logiciels, et on lui ajoute des métadonnées (informations complémentaires) nécessaires au plugin de QGIS.
- **Inférence dans QGIS** : le plugin Deepness charge le modèle ONNX, l'applique sur une nouvelle orthophoto et produit les couches vectorielles de détection.

Les sections qui suivent détaillent chacune de ces étapes, en commençant par les notions théoriques indispensables.

---

## 2. Les bases : qu'est-ce que la détection d'objets ?

### Classifier, localiser, détecter

En vision par ordinateur (la discipline qui apprend aux machines à « voir » et à comprendre des images), il existe plusieurs niveaux de compréhension d'une image :

- La **classification** répond à la question « que contient cette image ? ». Par exemple : « cette image contient des palmiers ». On obtient une étiquette globale, mais aucune information sur l'emplacement des objets.
- La **localisation** ajoute la question « où se trouve l'objet ? ». On dessine un rectangle (appelé **boîte englobante** ou *bounding box* en anglais) autour de l'objet principal.
- La **détection d'objets** combine les deux, et ce pour *tous* les objets présents dans l'image, pas seulement un seul. Le modèle de détection prend une image en entrée et produit en sortie une liste de résultats, où chaque résultat contient trois informations : la position de l'objet (sous forme de boîte englobante), sa classe (palmier ou maison) et un score de confiance (un nombre entre 0 et 1 indiquant la certitude du modèle).

Dans notre projet, c'est bien de la détection d'objets que nous faisons : sur chaque image, il peut y avoir zéro, un, dix ou cent objets, et le modèle doit tous les trouver, les localiser et les classifier.

### La boîte englobante

La boîte englobante est le concept central de la détection. C'est un rectangle qui entoure un objet de manière aussi ajustée que possible. En pratique, on ne stocke pas les quatre coins du rectangle. Dans le format utilisé par YOLO, on enregistre quatre nombres :

- **x_centre** : la position horizontale du centre de la boîte, exprimée en proportion de la largeur de l'image (entre 0 et 1). Si la valeur est 0.5, le centre est exactement au milieu de l'image en largeur.
- **y_centre** : la position verticale du centre, en proportion de la hauteur (entre 0 et 1). Si la valeur est 0.3, le centre est à 30 % du haut de l'image.
- **largeur** : la largeur de la boîte, en proportion de la largeur de l'image. Une valeur de 0.1 signifie que la boîte fait 10 % de la largeur de l'image.
- **hauteur** : la hauteur de la boîte, en proportion de la hauteur de l'image.

Ces valeurs sont dites **normalisées** : elles ne dépendent pas de la taille en pixels de l'image. Cela permet d'utiliser les mêmes annotations même si l'image est redimensionnée (par exemple de 680×680 à 640×640 pixels). Pour retrouver les coordonnées en pixels, il suffit de multiplier par la largeur ou la hauteur réelle de l'image. Par exemple, si l'image fait 640 pixels de large et que x_centre vaut 0.5, le centre est à 640 × 0.5 = 320 pixels du bord gauche.

### La confiance et le seuil

Quand le modèle analyse une image, il ne dit jamais « je suis certain à 100 % qu'il y a un palmier ici ». Il attribue un **score de confiance** : par exemple 0.92 (très sûr), 0.65 (assez sûr) ou 0.15 (peu sûr). On choisit un **seuil de confiance** (par exemple 0.3) en dessous duquel on rejette les détections. Si le score est de 0.15 et le seuil à 0.3, la détection est ignorée. Ce seuil permet de contrôler le compromis entre « trouver beaucoup d'objets » (seuil bas) et « ne garder que les détections fiables » (seuil haut).

---

## 3. L'apprentissage supervisé : comment un modèle apprend

### Le principe général

Un réseau de neurones, au départ, ne sait rien détecter. C'est un programme dont les millions de paramètres internes (appelés **poids**) sont initialisés avec des valeurs aléatoires. Pour qu'il apprenne, on utilise l'**apprentissage supervisé** : on lui montre des exemples accompagnés des bonnes réponses (les annotations faites par un humain), et on ajuste progressivement ses poids pour qu'il se rapproche de ces bonnes réponses.

Concrètement, voici ce qui se passe à chaque étape d'entraînement :

- Le modèle reçoit une image.
- Il produit des prédictions : des boîtes englobantes avec des classes et des scores.
- On compare ces prédictions aux annotations humaines (les « vraies » boîtes et classes). L'écart entre les prédictions et la réalité est mesuré par une **fonction de perte** (*loss function*), un nombre qui quantifie « à quel point le modèle s'est trompé ».
- Un algorithme appelé **rétropropagation du gradient** calcule dans quel sens et de combien il faut modifier chaque poids du réseau pour réduire cette erreur.
- Les poids sont mis à jour, et on recommence avec l'image suivante.

Un passage complet sur toutes les images d'entraînement s'appelle une **époque** (*epoch*). Au fil des époques, les prédictions du modèle se rapprochent de plus en plus des annotations. Après suffisamment d'époques, le modèle a « appris » à reconnaître les formes, les textures et les contextes propres aux palmiers et aux maisons vus du ciel, et il peut les détecter dans des images qu'il n'a jamais rencontrées.

### Le risque de sur-apprentissage

Un danger fréquent est le **sur-apprentissage** (ou *overfitting*) : le modèle finit par mémoriser les images d'entraînement au lieu d'apprendre des règles générales. Il obtient alors d'excellents résultats sur les données d'entraînement, mais des résultats médiocres sur de nouvelles images. C'est pour cela que l'on sépare les données en deux ensembles distincts (entraînement et validation) et que l'on surveille les performances sur l'ensemble de validation, qui contient des images que le modèle ne voit jamais pendant l'ajustement de ses poids.

### Le transfer learning (apprentissage par transfert)

Entraîner un modèle à partir de zéro nécessite des millions d'images. Avec seulement 110 tuiles annotées, ce serait insuffisant. La solution est le **transfer learning** : on part d'un modèle déjà entraîné sur un immense jeu de données généraliste (le dataset COCO, qui contient 330 000 images et 80 classes d'objets du quotidien : personnes, voitures, chiens, etc.). Ce modèle a déjà appris à reconnaître des formes générales — des bords, des textures, des contours. On conserve ces connaissances de base et on n'ajuste que les dernières couches du réseau pour qu'il apprenne nos deux classes spécifiques (palmier et maison). Cette technique s'appelle le **fine-tuning** (ajustement fin). Elle réduit considérablement le nombre d'images nécessaires et le temps d'entraînement.

---

## 4. YOLO : fonctionnement, historique et variantes

### Qu'est-ce que YOLO ?

YOLO est l'acronyme de « **You Only Look Once** » (« tu ne regardes qu'une fois »). C'est une famille d'algorithmes de détection d'objets particulièrement rapides. L'idée fondatrice est la suivante : au lieu de balayer l'image avec une loupe virtuelle à plusieurs endroits et à plusieurs échelles (ce que faisaient les méthodes antérieures), YOLO traite l'image entière en un seul passage à travers le réseau de neurones. En sortie, il produit directement toutes les boîtes englobantes et toutes les classes en une seule opération.

Pour comprendre intuitivement, imaginez que l'on superpose une grille invisible sur l'image (par exemple 20 lignes × 20 colonnes). Chaque cellule de cette grille est responsable de détecter les objets dont le centre tombe dans cette cellule. Pour chaque cellule, le réseau prédit un nombre fixe de boîtes possibles accompagnées de leurs classes et de leurs scores de confiance. Ensuite, un algorithme de post-traitement appelé **NMS** (*Non-Maximum Suppression*, suppression des non-maximums) élimine les boîtes redondantes (quand plusieurs cellules détectent le même objet) en ne conservant que celle avec le score le plus élevé.

### Les versions de YOLO

YOLO a évolué considérablement depuis sa création :

| Version | Année | Apport principal |
|---------|-------|------------------|
| YOLOv1 | 2016 | Introduction de la détection en un seul passage |
| YOLOv2 | 2017 | Ajout des « ancres » (formes de boîtes prédéfinies) pour améliorer la localisation |
| YOLOv3 | 2018 | Détection multi-échelle : le réseau travaille à trois résolutions différentes pour détecter les objets petits, moyens et grands |
| YOLOv4 | 2020 | Nouvelles techniques d'entraînement (CSP, PANet) pour une meilleure précision |
| YOLOv5 | 2020 | Implémentation en PyTorch par Ultralytics, très facile à utiliser et à déployer |
| YOLOv8 | 2023 | Architecture modernisée, API simplifiée, export ONNX natif |

Dans ce projet, nous utilisons **YOLOv8**, développé par la société Ultralytics. Il a été choisi parce qu'il est simple à installer (une seule commande `pip install ultralytics`), que son API Python est claire (`model.train()`, `model.val()`, `model.export()`), et qu'il exporte nativement au format ONNX, indispensable pour le plugin Deepness de QGIS.

### Les tailles de modèle YOLOv8

YOLOv8 n'est pas un modèle unique, mais une famille de cinq modèles de tailles différentes, identifiés par une lettre :

| Variante | Nom complet | Nombre de paramètres | Caractéristique |
|----------|-------------|---------------------|-----------------|
| **n** | nano | ~3 millions | Très léger, très rapide, adapté aux petits datasets |
| **s** | small | ~11 millions | Bon compromis pour des datasets moyens |
| **m** | medium | ~26 millions | Plus précis, demande plus de données |
| **l** | large | ~44 millions | Haute précision, nécessite un gros dataset |
| **x** | extra-large | ~68 millions | Maximum de précision, très gourmand en ressources |

Plus un modèle a de paramètres (c'est-à-dire de « neurones »), plus il peut apprendre de subtilités, mais plus il a besoin de données pour ne pas sur-apprendre. Avec nos ~110 images et 2 classes, le modèle **nano (YOLOv8n)** est le choix approprié. Il offre suffisamment de capacité pour distinguer des palmiers et des maisons vus du ciel, tout en limitant le risque d'overfitting. Les résultats obtenus (une mAP@50 d'environ 0.85) confirment que ce choix est pertinent.

### Le format des labels YOLO

Le format des annotations (ou « labels ») utilisé par YOLO est un format texte très simple. Pour chaque image (par exemple `mutwanga_tile_42.tif`), il existe un fichier texte du même nom (`mutwanga_tile_42.txt`). Chaque ligne de ce fichier décrit un objet et contient exactement cinq valeurs séparées par des espaces :

```
<classe> <x_centre> <y_centre> <largeur> <hauteur>
```

Par exemple :

```
0 0.5 0.3 0.1 0.08
1 0.72 0.61 0.15 0.12
```

La première ligne décrit un palmier (classe 0) centré à 50 % de la largeur et 30 % de la hauteur, avec une boîte faisant 10 % de la largeur et 8 % de la hauteur de l'image. La deuxième décrit une maison (classe 1). Si une image ne contient aucun objet, le fichier .txt est vide (ou absent).

---

## 5. Préparation des données : découpe des orthophotos en tuiles

### Pourquoi découper ?

Les orthophotos sont des images aériennes géoréférencées, souvent au format GeoTIFF. Le terme « géoréférencé » signifie que chaque pixel de l'image est rattaché à des coordonnées terrestres précises (latitude, longitude ou coordonnées projetées). Ces fichiers sont souvent énormes : des dizaines de milliers de pixels de côté et des centaines de mégaoctets, voire plusieurs gigaoctets.

Il est impossible de travailler directement avec ces images géantes pour trois raisons :

- **L'annotation** : le logiciel LabelImg ne peut pas charger confortablement des images de plusieurs gigaoctets, et l'annotateur (vous) ne peut pas visualiser des milliers de palmiers d'un coup.
- **L'entraînement** : le réseau de neurones attend des images de taille fixe (640×640 pixels dans notre cas). Lui envoyer une image de 30 000×30 000 pixels dépasserait la mémoire du GPU.
- **La quantité de données** : un modèle apprend mieux quand il dispose de nombreux exemples variés. Une seule grande image donne un seul exemple ; découpée en 100 tuiles, elle en donne cent.

### Comment découper

La découpe consiste à quadriller l'orthophoto en morceaux réguliers, comme un damier. Chaque morceau, appelé **tuile**, devient une image indépendante. Dans ce projet, les tuiles mesurent environ 680×680 pixels. Elles sont nommées de façon séquentielle : `mutwanga_tile_1.tif`, `mutwanga_tile_2.tif`, etc.

Cette découpe peut être réalisée de plusieurs façons :

- Avec **QGIS** : l'outil « Créer une grille » ou « Découper un raster selon une étendue » permet de diviser un raster en tuiles.
- Avec un **script Python** utilisant la bibliothèque `rasterio` (une bibliothèque pour lire et écrire des données raster géospatiales).
- Avec tout autre outil de tuilage géospatial (GDAL, etc.).

L'important est d'obtenir des tuiles de taille régulière et de les nommer de manière cohérente, car le fichier de labels (`.txt`) doit porter exactement le même nom que l'image (à l'extension près).

### La résolution spatiale

Un concept clé à comprendre est la **résolution spatiale**. Elle exprime la taille au sol d'un pixel. Par exemple, une résolution de 30 cm/px signifie qu'un pixel de l'image correspond à un carré de 30 cm × 30 cm au sol. Plus le chiffre est petit, plus l'image est détaillée : à 5 cm/px, on distingue des objets de quelques centimètres.

La résolution spatiale a un impact direct sur la détection. Les images d'entraînement de ce projet ont une résolution d'environ 30 cm/px. Un palmier vu à cette résolution occupe typiquement une boîte de quelques dizaines de pixels de côté. Si vous appliquez le modèle sur un raster à 11 cm/px (résolution plus fine), les mêmes palmiers apparaîtront plus grands en pixels. Le modèle peut toujours les détecter, mais il faut indiquer la bonne résolution dans les paramètres du modèle pour que les coordonnées des détections soient correctement converties en coordonnées terrain.

Dans notre projet, les tuiles sont stockées dans le dossier `D:\GIS\Recherche\Segmentation\Decoupe Raster\Mutwanga\2\`. C'est ce dossier qui sert de source à la fois pour l'annotation dans LabelImg et pour le script `prepare_dataset.py` qui alimente le dataset d'entraînement.

---

## 6. Annotation des images avec LabelImg

### Le rôle de l'annotation

L'annotation est l'étape la plus importante de tout projet de détection d'objets. C'est elle qui fournit au modèle les « bonnes réponses » à partir desquelles il apprend. Si les annotations sont mauvaises (boîtes mal placées, objets oubliés, classes inversées), le modèle apprendra mal, quels que soient les paramètres d'entraînement.

**LabelImg** est un logiciel gratuit et open source qui permet d'annoter des images pour la détection d'objets. Il affiche une image, vous laisse dessiner des rectangles autour des objets et associer chaque rectangle à une classe. Les annotations sont ensuite sauvegardées dans un fichier texte.

### Installation et configuration

L'installation de LabelImg demande un environnement Python avec quelques bibliothèques. Les étapes typiques sont :

- Créer un environnement Python (avec `venv` ou `conda`).
- Installer les dépendances : `PyQt5` (pour l'interface graphique) et `lxml` (pour la lecture de fichiers XML).
- Lancer le programme : `python labelImg.py`.

Avant de commencer à annoter, il faut configurer deux choses essentielles :

- Le **format de sauvegarde** : dans LabelImg, basculez le mode de sauvegarde sur « YOLO » (et non « PascalVOC »). Cela garantit que les fichiers .txt seront au format attendu.
- Le **fichier classes.txt** : ce fichier, placé dans le dossier des annotations ou dans le dossier `data/` de LabelImg, liste les noms des classes dans l'ordre. Dans notre cas, il ne contient que deux lignes :

```
palmier
maison
```

La première ligne correspond à la classe 0, la deuxième à la classe 1. Cet ordre doit être strictement respecté dans tout le projet.

### Bonnes pratiques d'annotation

La qualité du modèle est directement proportionnelle à la qualité des annotations. Voici les règles essentielles à suivre :

- **Ajuster la boîte au plus près de l'objet** : la boîte doit être aussi serrée que possible autour du palmier ou de la maison. Une boîte trop grande inclut du fond (herbe, route), ce qui brouille l'apprentissage. Une boîte trop petite coupe une partie de l'objet.
- **Annoter tous les objets de la classe** : si vous annotez les palmiers, vous devez annoter *tous* les palmiers visibles dans la tuile, pas seulement les plus évidents. Un palmier non annoté est perçu par le modèle comme « du fond » ; s'il détecte quand même un objet à cet endroit, il sera pénalisé pour un faux positif, ce qui est contre-productif.
- **Être cohérent** : si vous décidez de ne pas annoter les palmiers partiellement coupés par le bord de l'image, appliquez cette règle à toutes les tuiles. La cohérence est plus importante que la règle elle-même.
- **Vérifier les classes** : assurez-vous que le bon label est attribué au bon objet. Confondre un palmier et une maison pendant l'annotation entraîne le modèle à faire la même erreur.
- **Supprimer les classes parasites** : si LabelImg propose des classes par défaut (dog, person, etc.), supprimez-les du fichier `predefined_classes.txt` pour éviter les erreurs.

### Gestion des anciennes annotations

Il peut arriver que d'anciennes annotations utilisent d'autres numéros de classe (par exemple 15 pour palmier et 16 pour maison au lieu de 0 et 1). Le script `remap_labels.py` (décrit plus loin) permet de corriger automatiquement tous les fichiers `.txt` pour renuméroter les classes.

---

## 7. Organisation du dataset et fichier de configuration

### Pourquoi séparer entraînement et validation ?

Quand un modèle s'entraîne, il ajuste ses poids pour minimiser l'erreur sur les images qu'il voit. Mais on veut savoir s'il est capable de détecter des objets sur des images *qu'il n'a jamais vues*. C'est le rôle de l'ensemble de **validation** : un sous-ensemble d'images qui n'est jamais utilisé pour modifier les poids du modèle. On calcule les métriques (précision, rappel, mAP) uniquement sur cet ensemble. Si le modèle performe bien en validation, on peut avoir confiance dans sa capacité à travailler sur de nouvelles orthophotos.

La répartition standard est **80 % pour l'entraînement et 20 % pour la validation**. Les images sont mélangées aléatoirement avant la répartition, avec une graine fixe (un nombre qui initialise le générateur aléatoire, par exemple 42) pour que le même tirage soit reproductible d'une exécution à l'autre.

### La structure de dossiers

Le dataset doit respecter une arborescence précise, que YOLO reconnaît automatiquement :

```
dataset_palmiers/
├── images/
│   ├── train/          ← images d'entraînement (.tif ou .png)
│   └── val/            ← images de validation
├── labels/
│   ├── train/          ← labels d'entraînement (.txt, format YOLO)
│   └── val/            ← labels de validation
└── palms.yaml          ← fichier de configuration du dataset
```

Chaque image d'entraînement dans `images/train/` (par exemple `mutwanga_tile_42.tif`) doit avoir un fichier de labels correspondant dans `labels/train/` (par exemple `mutwanga_tile_42.txt`), portant exactement le même nom de base. Le fichier de labels contient une ligne par objet annoté dans l'image. La même correspondance existe pour les images de validation dans `images/val/` et `labels/val/`.

Dans ce projet, le dataset contient environ 110 images au total, soit environ 88 pour l'entraînement et 22 pour la validation. Le nombre total d'annotations est d'environ 6 857 objets (environ 4 700 palmiers et 2 100 maisons).

### Le fichier palms.yaml

Le fichier `palms.yaml` est le point d'entrée que YOLO lit pour savoir où sont les données et quelles sont les classes. Son contenu est simple :

```yaml
path: D:\GIS\Recherche\dataset_palmiers
train: images/train
val: images/val

nc: 2
names: ['palmier', 'maison']
```

- **path** : le chemin absolu vers le dossier racine du dataset. Ce chemin change selon l'environnement : en local sur Windows, c'est par exemple `D:\GIS\Recherche\dataset_palmiers` ; sur Google Colab, c'est `/content/drive/MyDrive/Recherche/dataset_palmiers`. Le notebook d'entraînement met automatiquement à jour ce champ.
- **train** et **val** : les chemins relatifs (par rapport à `path`) vers les dossiers d'images d'entraînement et de validation.
- **nc** : le nombre de classes (2 dans notre cas).
- **names** : la liste ordonnée des noms de classes. L'ordre doit correspondre aux numéros utilisés dans les fichiers de labels : la classe 0 est « palmier », la classe 1 est « maison ».

Les champs `nc` et `names` ne doivent jamais être modifiés d'un entraînement à l'autre (sauf si vous changez les classes du projet). Seul le champ `path` doit être adapté à l'environnement.

---

## 8. Scripts du projet : prepare_dataset et remap_labels

### Le script prepare_dataset.py

Ce script automatise la construction du dataset à partir du dossier d'annotation. Il évite de devoir copier manuellement les fichiers dans les bons sous-dossiers.

Voici ce qu'il fait, étape par étape :

1. Il parcourt le dossier source (par exemple `D:\GIS\Recherche\Segmentation\Decoupe Raster\Mutwanga\2\`) et recense toutes les tuiles qui possèdent un fichier `.txt` associé (les tuiles non annotées sont ignorées).
2. Il trie les tuiles par numéro, puis les mélange aléatoirement avec la graine fixe 42.
3. Il répartit les tuiles : 80 % dans l'entraînement, 20 % dans la validation.
4. Il vide les dossiers de destination (`images/train`, `images/val`, `labels/train`, `labels/val`) pour repartir de zéro et éviter de mélanger d'anciennes données avec les nouvelles.
5. Il copie chaque image `.tif` et son label `.txt` dans le bon sous-dossier.
6. Il affiche le nombre de tuiles par ensemble.

Ce script doit être relancé à chaque fois que de nouvelles tuiles ont été annotées. Sa configuration se résume à deux variables au début du fichier :

```python
SRC_DIR = r"D:\GIS\Recherche\Segmentation\Decoupe Raster\Mutwanga\2"
DST_DIR = r"D:\GIS\Recherche\dataset_palmiers"
```

Adaptez ces chemins à votre machine si nécessaire.

### Le script remap_labels.py

Ce script résout un problème très spécifique : quand les annotations ont été faites avec de « mauvais » numéros de classe. Par exemple, si à cause d'une configuration initiale de LabelImg les palmiers ont été enregistrés avec la classe 15 et les maisons avec la classe 16 au lieu de 0 et 1, il faut corriger tous les fichiers .txt avant de les utiliser pour l'entraînement.

Le script parcourt chaque fichier `.txt` du dossier d'annotation et, pour chaque ligne :

- Si la classe est 15, il la remplace par 0 (palmier).
- Si la classe est 16, il la remplace par 1 (maison).
- Sinon, il supprime la ligne (classe parasite).

Il réécrit ensuite le fichier avec les lignes corrigées. Ce script ne doit être exécuté qu'une seule fois, avant de lancer `prepare_dataset.py`.

---

## 9. Le notebook d'entraînement sur Google Colab

### Pourquoi Google Colab ?

L'entraînement d'un modèle de deep learning nécessite un **GPU** (processeur graphique), une puce conçue pour effectuer des calculs massivement parallèles. Sans GPU, l'entraînement qui prend 30 minutes pourrait durer des heures, voire des jours. Google Colab est un service gratuit qui met à disposition un GPU dans le cloud. On écrit et exécute du code Python dans un « notebook » (un document interactif composé de cellules de code et de texte) directement dans le navigateur web.

### Le notebook train_palmiers_maisons_yolov8_deepness_v2.ipynb

Le notebook d'entraînement est organisé en étapes logiques. Voici ce que fait chaque section :

**Étape 1 — Monter Google Drive.** Le dataset est stocké dans votre Google Drive. La première cellule du notebook connecte Colab à votre Drive pour accéder aux fichiers.

**Étape 2 — Installer les dépendances.** On installe deux bibliothèques Python :
- `ultralytics` : la bibliothèque officielle de YOLOv8, qui fournit tout le code d'entraînement, de validation et d'export.
- `onnx` : une bibliothèque pour manipuler les fichiers au format ONNX (utilisé pour l'export du modèle).

**Étape 3 — Configurer le chemin du dataset.** Le notebook cherche automatiquement le dossier `dataset_palmiers` dans votre Google Drive et met à jour le champ `path` du fichier `palms.yaml`. Il vérifie aussi que le fichier est correctement configuré (nc=2, noms des classes) et compte les images et labels dans chaque sous-dossier.

**Étape 4 — Diagnostiquer le dataset.** Avant d'entraîner, le notebook analyse le dataset pour détecter d'éventuels problèmes : labels manquants ou vides, distribution des classes (y a-t-il beaucoup plus de palmiers que de maisons ?), taille des boîtes englobantes (y a-t-il des boîtes anormalement petites ou grandes ?), et nombre d'objets par image. Des graphiques sont affichés pour visualiser ces statistiques.

**Étape 5 — Nettoyer les labels.** Certains fichiers texte peuvent contenir un caractère invisible appelé **BOM** (*Byte Order Mark*, un marqueur d'encodage ajouté automatiquement par certains éditeurs de texte sous Windows). Ce caractère parasite empêche YOLO de lire correctement la première ligne du fichier. Le notebook le supprime automatiquement.

**Étape 6 — Convertir les images.** Les GeoTIFF ont souvent 4 canaux de couleur (RGBA : rouge, vert, bleu et alpha/transparence), mais YOLOv8 attend 3 canaux (RGB). Le notebook convertit toutes les images en PNG à 3 canaux et supprime les anciens fichiers .tif pour éviter les doublons.

**Étape 7 — Entraîner le modèle.** C'est la cellule centrale du notebook. Elle charge un modèle YOLOv8n pré-entraîné et lance l'entraînement avec tous les paramètres détaillés dans la section suivante.

**Étapes 8 et 9 — Visualiser et évaluer.** Le notebook affiche les courbes de perte (*loss*), les métriques de validation (précision, rappel, mAP) et la matrice de confusion. Il calcule aussi les métriques par classe (palmier séparément, maison séparément).

**Étapes 10 à 12 — Exporter en ONNX et ajouter les métadonnées.** Le meilleur modèle est converti au format ONNX, enrichi de métadonnées Deepness, puis sauvegardé dans Google Drive.

---

## 10. Paramètres d'entraînement expliqués un par un

Quand on lance l'entraînement avec `model.train(...)`, on passe de nombreux paramètres. Chacun contrôle un aspect de l'apprentissage. Voici leur rôle, expliqué de façon aussi claire que possible.

### Paramètres de base

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `data` | `palms.yaml` | Indique au modèle où trouver les images, les labels et les noms des classes. |
| `epochs` | 200 | Nombre maximum de passages complets sur les données d'entraînement. |
| `patience` | 30 | Si la performance en validation ne s'améliore pas pendant 30 époques consécutives, l'entraînement s'arrête automatiquement. On parle d'**arrêt anticipé** (*early stopping*). Cela évite de gaspiller du temps de calcul et de sur-apprendre. |
| `imgsz` | 640 | Taille en pixels à laquelle chaque image est redimensionnée avant d'être envoyée au réseau. Nos tuiles font environ 680 pixels ; 640 est la valeur standard la plus proche. |
| `batch` | 8 | Nombre d'images traitées ensemble à chaque étape. Le réseau calcule l'erreur moyenne sur ces 8 images avant de mettre à jour ses poids. Un batch plus grand stabilise l'apprentissage mais consomme plus de mémoire GPU. |

### Paramètres de fine-tuning

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `freeze` | 10 | « Gèle » les 10 premières couches du réseau (le **backbone**, la partie qui extrait des caractéristiques générales de l'image). Ces couches conservent les connaissances acquises sur COCO et ne sont pas modifiées. Seules les couches suivantes (la « tête » de détection) sont entraînées. Cela réduit le risque d'overfitting avec un petit dataset. |
| `lr0` | 0.001 | **Learning rate** (taux d'apprentissage) initial. C'est la taille du « pas » de mise à jour des poids à chaque étape. Un pas trop grand fait osciller le modèle et l'empêche de converger ; un pas trop petit ralentit l'apprentissage. Pour du fine-tuning, on utilise une valeur modérée. |
| `lrf` | 0.01 | Facteur de réduction du learning rate en fin d'entraînement. Le LR final sera lr0 × lrf = 0.001 × 0.01 = 0.00001. L'idée est de faire de gros pas au début (pour apprendre vite) et de petits pas à la fin (pour affiner la convergence). |
| `cos_lr` | True | Active la réduction du learning rate selon une courbe en cosinus (une descente douce et progressive) au lieu d'une réduction par paliers. |

### Paramètres d'augmentation de données

L'**augmentation de données** (*data augmentation*) consiste à appliquer des transformations aléatoires aux images pendant l'entraînement pour créer de la diversité artificielle. Le modèle voit la même image sous des angles, des orientations et des échelles différents, ce qui l'aide à généraliser.

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `degrees` | 90.0 | Rotation aléatoire de l'image jusqu'à ±90°. Les images aériennes sont souvent prises à la verticale (vue zénithale) ; un palmier ou une maison a la même apparence tourné de 90°, donc cette augmentation est très pertinente. |
| `flipud` | 0.5 | Retournement vertical avec une probabilité de 50 %. Encore une fois, en vue aérienne, un retournement vertical ne change pas la nature des objets. |
| `fliplr` | 0.5 | Retournement horizontal (miroir) avec une probabilité de 50 %. |
| `scale` | 0.3 | Variation aléatoire de l'échelle (zoom) de ±30 %. Le modèle voit les objets à des tailles légèrement différentes, ce qui le rend plus robuste aux variations de résolution. |
| `mosaic` | 1.0 | Active à 100 % la technique du **mosaïque** : quatre images sont assemblées en une seule pendant l'entraînement. Le modèle voit ainsi davantage de contexte et d'objets par image. |
| `close_mosaic` | 20 | Désactive le mosaïque pendant les 20 dernières époques pour stabiliser la convergence finale. |
| `copy_paste` | 0.3 | Copie-colle aléatoirement des objets d'une image à une autre. Cela crée des scènes artificielles plus denses et aide le modèle à apprendre à détecter des objets dans des contextes variés. |
| `mixup` | 0.15 | Superpose deux images avec une certaine transparence. C'est une forme de régularisation qui réduit le risque d'overfitting. |

---

## 11. Comprendre les métriques : précision, rappel, mAP

À la fin de l'entraînement, et à chaque époque de validation, le modèle est évalué avec des métriques standardisées. Comprendre ces métriques est essentiel pour savoir si le modèle est bon et, si non, dans quelle direction l'améliorer.

### Précision (Precision)

La **précision** répond à la question : **parmi toutes les détections proposées par le modèle, combien sont correctes ?**

$$\text{Précision} = \frac{\text{Vrais positifs}}{\text{Vrais positifs} + \text{Faux positifs}}$$

Un « vrai positif » est une détection qui correspond bien à un vrai objet (le modèle a raison). Un « faux positif » est une détection qui ne correspond à rien (le modèle a « inventé » un objet ou s'est trompé de classe).

Par exemple, si le modèle propose 100 détections et que 85 sont correctes, la précision est $\frac{85}{100} = 0.85$ soit 85 %. Une précision basse indique que le modèle « hallucine » trop de faux objets. On peut alors augmenter le seuil de confiance pour ne garder que les détections les plus sûres.

### Rappel (Recall)

Le **rappel** répond à la question : **parmi tous les vrais objets présents dans les images, combien le modèle a-t-il retrouvés ?**

$$\text{Rappel} = \frac{\text{Vrais positifs}}{\text{Vrais positifs} + \text{Faux négatifs}}$$

Un « faux négatif » est un vrai objet que le modèle a raté. Si les annotations indiquent 200 palmiers et que le modèle en trouve 160, le rappel est $\frac{160}{200} = 0.80$ soit 80 %. Un rappel bas signifie que le modèle passe à côté de beaucoup d'objets. On peut alors baisser le seuil de confiance (au risque d'augmenter les faux positifs) ou améliorer le modèle.

### IoU (Intersection over Union)

Pour décider si une détection « correspond » bien à un vrai objet, on utilise l'**IoU** (*Intersection over Union*, intersection sur union). C'est le rapport entre la surface de chevauchement des deux boîtes (prédite et réelle) et la surface de leur union. Si l'IoU est élevé (proche de 1), les deux boîtes se recouvrent presque parfaitement. Si l'IoU est faible (proche de 0), elles ne se chevauchent presque pas.

$$\text{IoU} = \frac{\text{Surface d'intersection}}{\text{Surface d'union}}$$

On fixe un seuil d'IoU (par exemple 0.50) : si l'IoU entre la boîte prédite et une boîte réelle dépasse ce seuil, la détection est considérée comme correcte (vrai positif). Sinon, elle est comptée comme faux positif.

### mAP (mean Average Precision)

La **mAP** est la métrique de référence en détection d'objets. Elle agrège la précision et le rappel de façon élégante :

1. Pour chaque classe, on trie les détections par score de confiance décroissant.
2. On calcule la précision et le rappel cumulés au fur et à mesure qu'on ajoute les détections.
3. On obtient une courbe précision-rappel pour chaque classe.
4. L'aire sous cette courbe est l'**AP** (*Average Precision*) de la classe.
5. La **mAP** est la moyenne des AP sur toutes les classes.

Deux variantes sont couramment utilisées :

- **mAP@50** : calculée avec un seuil IoU de 0.50. C'est la métrique la plus couramment citée. Un mAP@50 de 0.85 signifie que le modèle détecte correctement 85 % des objets (en combinant précision et rappel) lorsqu'on accepte des boîtes qui chevauchent la réalité à au moins 50 %.
- **mAP@50-95** : moyenne des mAP calculées pour des seuils IoU allant de 0.50 à 0.95 par pas de 0.05. C'est une métrique plus stricte qui exige des boîtes très bien positionnées.

### La matrice de confusion

La **matrice de confusion** est un tableau qui montre comment les prédictions se répartissent par rapport à la réalité :

|  | Prédit : palmier | Prédit : maison | Prédit : fond |
|--|-----------------|----------------|---------------|
| **Vrai : palmier** | vrai positif | confusion | faux négatif |
| **Vrai : maison** | confusion | vrai positif | faux négatif |

Les cases sur la diagonale (où vrai = prédit) sont les bonnes réponses. Les cases hors diagonale montrent les erreurs. Par exemple, si le modèle confond souvent des maisons avec des palmiers, la case (Vrai : maison, Prédit : palmier) aura une valeur élevée.

### Les courbes de loss

Pendant l'entraînement, le notebook affiche des courbes de **loss** (perte/erreur) pour l'entraînement et la validation. La loss d'entraînement doit diminuer régulièrement. La loss de validation doit aussi diminuer, mais elle peut commencer à remonter si le modèle sur-apprend. L'écart entre les deux courbes est un indicateur de sur-apprentissage : si la loss d'entraînement est beaucoup plus basse que la loss de validation, le modèle a commencé à mémoriser les images d'entraînement au lieu de généraliser.

Le fichier `weights/best.pt`, sauvegardé automatiquement pendant l'entraînement, contient les poids du modèle à l'époque où la mAP de validation était la meilleure.

---

## 12. Export du modèle au format ONNX

### Qu'est-ce qu'ONNX ?

**ONNX** (*Open Neural Network Exchange*) est un format de fichier ouvert et standardisé pour stocker des modèles de deep learning. L'avantage principal d'ONNX est la **portabilité** : un modèle entraîné avec PyTorch (la bibliothèque Python utilisée par YOLOv8) peut être exporté en ONNX et ensuite exécuté avec un moteur d'exécution différent, comme ONNX Runtime, TensorRT ou le plugin Deepness de QGIS. On n'a plus besoin de PyTorch ni d'Ultralytics pour faire fonctionner le modèle.

### Comment exporter

L'export est simple : on charge le meilleur modèle et on appelle la méthode `export()` :

```python
from ultralytics import YOLO

best_model = YOLO('chemin/vers/best.pt')
onnx_path = best_model.export(format='onnx', imgsz=640, opset=17)
```

- `format='onnx'` : le format de sortie.
- `imgsz=640` : la taille d'entrée du modèle. Doit être identique à celle utilisée pendant l'entraînement.
- `opset=17` : la version du jeu d'opérations (*operator set*) ONNX. L'opset 17 assure la compatibilité avec Deepness.

Le fichier produit (par exemple `best.onnx`) contient à la fois l'architecture du réseau et tous ses poids.

---

## 13. Ajout des métadonnées Deepness

### Pourquoi des métadonnées ?

Le fichier ONNX brut contient le réseau de neurones, mais rien qui indique au plugin Deepness comment l'utiliser. Deepness a besoin de savoir :

- Quel **type de modèle** c'est (détection, segmentation, classification).
- Quels sont les **noms des classes** et leurs numéros.
- Quelle est la **résolution spatiale** de référence (en cm par pixel).
- Quels **seuils** appliquer (confiance minimale, IoU pour le NMS).
- Quel **format de sortie** le modèle utilise (YOLO Ultralytics, YOLO classique, etc.).

Ces informations sont stockées sous forme de paires clé-valeur directement dans le fichier ONNX, dans une zone prévue à cet effet (les `metadata_props`).

### Les métadonnées à ajouter

Voici les métadonnées ajoutées dans notre notebook :

| Clé | Valeur | Signification |
|-----|--------|---------------|
| `model_type` | `"Detector"` | Indique que c'est un modèle de détection d'objets (et non de segmentation ou de classification). |
| `class_names` | `{"0": "palmier", "1": "maison"}` | Dictionnaire des classes et de leurs noms. Deepness l'utilise pour créer des couches vectorielles nommées. |
| `resolution` | `30` | Résolution en cm/pixel. **Important** : cette valeur doit correspondre à la résolution du raster sur lequel vous ferez l'inférence. Si le raster est à 11 cm/px, mettez 11. |
| `det_conf` | `0.3` | Seuil de confiance minimum : les détections avec un score inférieur à 0.3 sont ignorées. |
| `det_iou_thresh` | `0.5` | Seuil IoU pour le NMS : si deux boîtes se chevauchent avec un IoU > 0.5, seule la meilleure est conservée. |
| `det_type` | `"YOLO_Ultralytics"` | Indique à Deepness quel post-traitement appliquer aux sorties du réseau (format spécifique à Ultralytics). |

### Attention à l'encodage JSON

Deepness décode chaque valeur de métadonnée avec `json.loads()`. Cela signifie que chaque valeur doit être une chaîne JSON valide. C'est pour cela qu'on utilise `json.dumps()` pour encoder chaque valeur, y compris les nombres et les dictionnaires :

```python
import json

m.key = 'resolution'
m.value = json.dumps(30)        # produit la chaîne "30"

m.key = 'class_names'
m.value = json.dumps({0: 'palmier', 1: 'maison'})
# produit la chaîne '{"0": "palmier", "1": "maison"}'
```

Il faut aussi supprimer les métadonnées ajoutées par défaut par Ultralytics lors de l'export, car elles ne sont pas au format JSON valide attendu par Deepness. Le notebook le fait automatiquement avant d'ajouter les nouvelles métadonnées.

Le modèle final est sauvegardé sous le nom `palmier_maison_yolov8n_deepness.onnx`.

---

## 14. Utilisation dans QGIS avec le plugin Deepness

### Installation du plugin

**QGIS** est un logiciel libre de Système d'Information Géographique (SIG) qui permet de visualiser, éditer et analyser des données géographiques. Le plugin **Deepness** ajoute à QGIS la capacité d'exécuter des modèles de deep learning directement sur les rasters affichés.

Pour installer Deepness :

1. Ouvrez QGIS.
2. Allez dans le menu **Extensions** (ou **Plugins**) → **Gérer et installer des extensions**.
3. Recherchez « Deepness ».
4. Cliquez sur **Installer**.

### Lancer une détection

Une fois le plugin installé et votre orthophoto chargée dans QGIS (comme couche raster), voici les étapes :

1. Ouvrez l'outil Deepness via le menu **Plugins → Deepness → Detection**.
2. Sélectionnez le fichier du modèle : c'est le fichier `.onnx` (par exemple `palmier_maison_yolov8n_deepness.onnx`). Attention à ne pas sélectionner par erreur le notebook `.ipynb` ou un autre fichier ; cela provoquerait une erreur « Protobuf parsing failed ».
3. Deepness lit automatiquement les métadonnées du modèle (seuil de confiance, IoU, résolution, noms des classes) et pré-remplit les paramètres.
4. **Vérifiez la résolution** : c'est le point le plus critique. La résolution en cm/pixel doit correspondre à celle du raster que vous analysez. Si votre orthophoto est à 11 cm/px mais que le modèle indique 30, modifiez la valeur dans Deepness pour la mettre à 11. Sinon, les polygones de détection seront mal dimensionnés et mal positionnés.
5. Lancez l'inférence.

### Ce qui se passe en coulisses

Le plugin découpe automatiquement le raster en tuiles de la taille attendue par le modèle (640×640 pixels), applique le modèle sur chaque tuile, collecte toutes les détections, puis applique un NMS global pour fusionner les détections redondantes aux frontières des tuiles. Enfin, il convertit les coordonnées pixel en coordonnées géographiques et crée les couches vectorielles.

### Résultat

Après l'inférence, de nouvelles couches apparaissent dans le panneau des couches de QGIS. Chaque couche contient des polygones rectangulaires correspondant aux détections. Les attributs de chaque polygone incluent :

- La **classe** de l'objet (palmier ou maison).
- Le **score de confiance**.

Vous pouvez ensuite :

- **Styliser** les couches par classe (par exemple : vert pour les palmiers, rouge pour les maisons).
- **Filtrer** par score de confiance pour ne garder que les détections les plus fiables.
- **Exporter** les couches en GeoPackage, Shapefile ou autre format géospatial.
- **Compter** les objets par zone, calculer des densités, faire des analyses spatiales.

### Performances sur les gros rasters

Sur des rasters très volumineux (plusieurs dizaines de gigaoctets), l'inférence peut être longue et la phase de fusion des détections peut sembler bloquée (par exemple à 99 %). C'est normal : le plugin doit fusionner des millions de détections candidates. Pour de tels cas, une solution est de traiter le raster par zones ou d'utiliser un script dédié qui traite les tuiles de façon incrémentale. Pour des rasters de taille modérée (quelques centaines de mégaoctets), Deepness reste la solution la plus simple et la plus directe.

---

## 15. Résumé de la chaîne complète et conseils pratiques

### Récapitulatif des étapes

| Étape | Action | Outil | Résultat |
|-------|--------|-------|----------|
| 1 | Découpe de l'orthophoto en tuiles | QGIS / rasterio | Tuiles .tif de ~680×680 px |
| 2 | Annotation des tuiles | LabelImg | Fichiers .txt (format YOLO) |
| 3 | Correction des labels (si nécessaire) | remap_labels.py | Labels avec classes 0 et 1 |
| 4 | Construction du dataset train/val | prepare_dataset.py | Arborescence images/ + labels/ |
| 5 | Entraînement du modèle | Google Colab + notebook | best.pt (poids du modèle) |
| 6 | Export ONNX + métadonnées | Notebook (cellules dédiées) | palmier_maison_yolov8n_deepness.onnx |
| 7 | Détection sur de nouvelles images | QGIS + plugin Deepness | Couches vectorielles de détection |

### Conseils pour améliorer les résultats

- **Plus de données** : la meilleure façon d'améliorer un modèle est d'annoter plus d'images. Passez de 110 à 200 ou 300 tuiles annotées et la mAP augmentera significativement.
- **Qualité des annotations** : relisez vos annotations. Corrigez les boîtes mal positionnées et les objets oubliés. Une heure passée à corriger les labels peut valoir plus qu'une semaine à ajuster les hyperparamètres.
- **Équilibre des classes** : si une classe a beaucoup plus d'exemples que l'autre (ici : environ 4 700 palmiers contre 2 100 maisons), le modèle risque de mieux détecter la classe majoritaire. Essayez d'annoter davantage d'images riches en maisons.
- **Résolution cohérente** : assurez-vous que la résolution spatiale des images d'inférence est renseignée correctement dans les métadonnées ou dans les paramètres de Deepness.
- **Tester plusieurs seuils** : après l'inférence, filtrez les résultats avec différents seuils de confiance (0.2, 0.3, 0.5) pour trouver le meilleur compromis entre faux positifs et faux négatifs pour votre usage.

---

## 16. Glossaire

| Terme | Définition |
|-------|------------|
| **Annotation** | Processus de marquage manuel des objets dans une image (dessiner des boîtes et indiquer les classes). |
| **Backbone** | Partie du réseau de neurones qui extrait les caractéristiques visuelles de l'image (formes, textures, couleurs). |
| **Batch** | Groupe d'images traitées ensemble à chaque étape d'entraînement. |
| **Bounding box** | Boîte englobante : rectangle qui entoure un objet détecté. |
| **Classe** | Catégorie d'objet (ici : palmier ou maison). |
| **CNN** | *Convolutional Neural Network* (réseau de neurones convolutif) : type de réseau spécialement adapté au traitement d'images. |
| **Dataset** | Jeu de données : ensemble structuré d'images et de labels utilisés pour l'entraînement et la validation. |
| **Deep learning** | Sous-domaine de l'intelligence artificielle utilisant des réseaux de neurones à plusieurs couches pour apprendre à partir de données. |
| **Époque** | Un passage complet sur toutes les images d'entraînement. |
| **Faux négatif** | Vrai objet que le modèle n'a pas détecté. |
| **Faux positif** | Détection proposée par le modèle qui ne correspond à aucun vrai objet. |
| **Fine-tuning** | Ajustement fin : adapter un modèle pré-entraîné à une tâche spécifique en n'entraînant que certaines couches. |
| **GeoTIFF** | Format d'image raster qui contient des informations de géoréférencement (coordonnées terrestres de chaque pixel). |
| **GPU** | *Graphics Processing Unit* : processeur graphique utilisé pour accélérer les calculs de deep learning. |
| **IoU** | *Intersection over Union* : mesure de chevauchement entre deux boîtes (0 = aucun chevauchement, 1 = superposition parfaite). |
| **Label** | Étiquette associée à un objet dans une annotation (ici : fichier .txt contenant les boîtes et les classes). |
| **Learning rate** | Taux d'apprentissage : contrôle la taille des ajustements des poids à chaque étape d'entraînement. |
| **Loss** | Fonction de perte : mesure numérique de l'erreur du modèle. Plus elle est basse, meilleures sont les prédictions. |
| **mAP** | *mean Average Precision* : métrique de référence en détection d'objets, combinant précision et rappel. |
| **NMS** | *Non-Maximum Suppression* : algorithme qui élimine les détections redondantes en ne conservant que la meilleure pour chaque objet. |
| **ONNX** | *Open Neural Network Exchange* : format ouvert et portable pour stocker des modèles de deep learning. |
| **Orthophoto** | Image aérienne rectifiée et géoréférencée, où chaque pixel correspond à une position au sol précise. |
| **Overfitting** | Sur-apprentissage : le modèle mémorise les données d'entraînement au lieu d'apprendre des règles générales. |
| **Précision** | Proportion de détections correctes parmi toutes les détections proposées par le modèle. |
| **Rappel** | Proportion d'objets réels retrouvés par le modèle parmi tous les objets présents. |
| **Raster** | Image composée d'une grille de pixels, par opposition aux données vectorielles (points, lignes, polygones). |
| **Réseau de neurones** | Programme informatique composé de couches de « neurones » artificiels capables d'apprendre des motifs dans les données. |
| **Résolution spatiale** | Taille au sol d'un pixel (en cm/px). Plus le nombre est petit, plus l'image est détaillée. |
| **SIG** | Système d'Information Géographique : logiciel pour manipuler, analyser et visualiser des données géographiques. |
| **Seuil de confiance** | Score minimum au-dessus duquel une détection est conservée. |
| **Transfer learning** | Apprentissage par transfert : réutiliser un modèle pré-entraîné sur une tâche différente comme point de départ. |
| **Tuile** | Morceau d'image obtenu par découpe d'un raster plus grand. |
| **Validation** | Ensemble d'images réservé à l'évaluation du modèle, jamais utilisé pour ajuster les poids. |
| **Vecteur** | Donnée géographique représentée par des formes géométriques (points, lignes, polygones) plutôt que par des pixels. |
| **YOLO** | *You Only Look Once* : famille d'algorithmes de détection d'objets rapides, traitant l'image en un seul passage. |

---

*Cours rédigé pour le projet de détection de palmiers et maisons sur orthophotos (YOLOv8 / Deepness / QGIS). Dataset : `dataset_palmiers`. Dernière mise à jour : février 2026.*
