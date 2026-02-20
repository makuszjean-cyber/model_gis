**Relabelling Workflow**

- **But :** collecter les faux positifs, relabeler les crops comme `other_tree` et injecter ces annotations dans le dataset YOLO pour améliorer la classe `palmier`.

**Scripts clés**
- [collect_false_positives.py](collect_false_positives.py): exécute le modèle sur un dossier d'images et sauve les crops détectés (CSV + images) pour revue.
- [apply_reviews_to_labels.py](apply_reviews_to_labels.py): prend `false_positives/false_positives.csv` + `review.csv` et ajoute des lignes YOLO (classe id configurable) dans `labels/`.
- [yolo_wrapper.py](yolo_wrapper.py): helpers pour charger le modèle et lancer l'inférence.

**Étapes recommandées (rapide)**
1. Active ton env et installe les dépendances:

```powershell
.\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Génère les crops (exemple sur le split de validation):

```powershell
python collect_false_positives.py --weights yolo26n.pt --source images/val --target-class 0 --conf 0.25 --outdir false_positives
```

3. Revue manuelle:
- Ouvre `false_positives/crops/` et marque chaque image dans un fichier `review.csv` (colonnes `crop,label`). Exemple:

```
crop,label
false_positives/crops/mutwanga_tile_12_det0_conf0.80.jpg,other_tree
false_positives/crops/mutwanga_tile_15_det1_conf0.70.jpg,keep
```

4. Applique les reviews aux labels YOLO (dry-run d'abord):

```powershell
python apply_reviews_to_labels.py --false-csv false_positives/false_positives.csv --review-csv review.csv --labels-root labels --label-id 2 --dry-run
```

Puis sans `--dry-run` pour écrire les fichiers:

```powershell
python apply_reviews_to_labels.py --false-csv false_positives/false_positives.csv --review-csv review.csv --labels-root labels --label-id 2
```

5. Mettre à jour la config de dataset si tu ajoutes la classe:
- Fichier: [palms.yaml](palms.yaml)
- Change `nc: 2` → `nc: 3` et `names: ['palmier','maison']` → `names: ['palmier','maison','other_tree']`.

6. Réentraîner / continuer l'entraînement (exemple Ultralytics CLI):

```powershell
yolo task=detect mode=train model=yolov8n.pt data=palms.yaml epochs=50 imgsz=640
```

**Conseils pratiques**
- Commence par un petit nombre de hard-negatives (quelques centaines) et réentraîne; observe l'impact sur precision/recall.
- Utilise `--dry-run` pour vérifier avant d'écrire dans `labels/`.
- Si beaucoup d'arbres non-palmier persistent, tu peux entraîner un classifieur de second niveau: detect→crop→classify.

**Fichiers créés par les scripts**
- `false_positives/crops/` : images crops à relabeler
- `false_positives/false_positives.csv` : metadata (image, crop, conf, class, bbox)

Si tu veux, j'applique automatiquement la modification de [palms.yaml](palms.yaml) (nc/names), ou je génère un `review.csv` template. Dis-moi quelle action tu veux faire ensuite.
