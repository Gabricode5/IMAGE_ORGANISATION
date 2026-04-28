# Journal des Incidents Techniques
## Application de Classification d'Images par Intelligence Artificielle

**Projet :** Classification d'Images IA — ResNet50  
**Auteur :** Gabriel Guery  
**Date du document :** 28 avril 2025  

---

## INCIDENT #001 — Erreur de dimensions dans le prétraitement de l'image

**Date de détection :** 28 avril 2025 à 10h14  
**Date de résolution :** 28 avril 2025 à 10h52  
**Durée de l'incident :** 38 minutes  
**Gravité :** Critique — l'API ne pouvait traiter aucune image  
**Statut :** ✅ Résolu  

---

### 1. Contexte

Dans le cadre du développement de l'API de classification d'images, un endpoint `POST /predict` a été créé dans le fichier `api.py`. Cet endpoint reçoit une image envoyée par un utilisateur, la transmet au modèle ResNet50, et retourne la classe prédite avec son score de confiance.

Lors de la première phase de tests, aucune prédiction n'a pu être effectuée. Toutes les requêtes envoyées à l'endpoint `/predict` retournaient une erreur HTTP 500 (Internal Server Error), rendant l'API totalement inutilisable.

---

### 2. Description de l'incident

Lors du premier test de l'API avec une image de chien (format JPG, 1200x800 pixels), la réponse reçue était la suivante :

```json
{
  "detail": "Internal Server Error"
}
```

En parallèle, le système de journalisation (`app.log`) enregistrait automatiquement l'erreur grâce au module Python `logging` mis en place dans le cadre du monitoring de l'application.

#### Extrait du fichier `app.log` au moment de l'incident

```
2025-04-28 10:14:32 — INFO — Initialisation du système de classification d'images...
2025-04-28 10:14:45 — INFO — Modèle ResNet50 chargé avec succès.
2025-04-28 10:14:51 — INFO — Fichier historique créé : predictions_history.csv
2025-04-28 10:15:03 — ERROR — Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:03 — ERROR — Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:03 — ERROR — Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:04 — CRITICAL — INCIDENT CRITIQUE — 3 erreurs consécutives ! Intervention requise.
```

Le système de détection automatique d'incidents a correctement déclenché une alerte de niveau `CRITICAL` après 3 erreurs consécutives, conformément aux seuils configurés dans `config.py`.

---

### 3. Analyse de la cause racine

#### 3.1 Fonctionnement attendu de ResNet50

Le modèle ResNet50 est un réseau de neurones convolutif entraîné sur le dataset ImageNet. Il impose une contrainte stricte sur la taille des images en entrée : **toutes les images doivent être redimensionnées à 224 × 224 pixels** avant d'être transmises au modèle.

Si une image d'une autre taille est fournie, TensorFlow/Keras lève immédiatement une exception car les dimensions ne correspondent pas aux couches internes du réseau.

#### 3.2 Identification du bug dans le code

En inspectant la fonction `preprocess_image()` dans `main.py`, la ligne responsable du redimensionnement était la suivante :

```python
# CODE BUGUÉ — à ne pas utiliser
img = img.resize((64, 64))
```

La valeur `(64, 64)` avait été utilisée par erreur lors du développement, probablement pour tester rapidement la vitesse de traitement avec des petites images. Cette valeur n'a pas été corrigée avant la mise en service de l'API.

#### 3.3 Chaîne de traitement et impact

Voici la chaîne de traitement d'une image dans l'application, avec le point de défaillance identifié :

```
Image envoyée par l'utilisateur (n'importe quelle taille)
        ↓
preprocess_image() — redimensionnement à (64, 64)  ← BUG ICI
        ↓
np.expand_dims() — ajout de la dimension batch → shape (1, 64, 64, 3)
        ↓
ResNet50.predict() — attend shape (1, 224, 224, 3)
        ↓
ERREUR : incompatibilité de dimensions → exception levée
        ↓
Logging automatique : ERROR puis CRITICAL
```

---

### 4. Correction apportée

#### 4.1 Modification du code

La correction consiste à remplacer la valeur codée en dur `(64, 64)` par la valeur définie dans le fichier de configuration `config.py`, via la clé `MODEL_CONFIG["input_shape"]`.

**Avant correction :**

```python
# INCORRECT — valeur codée en dur
img = img.resize((64, 64))
```

**Après correction :**

```python
# CORRECT — valeur lue depuis config.py
img = img.resize(MODEL_CONFIG["input_shape"])  # → (224, 224)
```

#### 4.2 Valeur dans `config.py`

```python
MODEL_CONFIG = {
    "model_name": "ResNet50",
    "weights": "imagenet",
    "input_shape": (224, 224),   # ← taille correcte pour ResNet50
    "top_k": 5
}
```

#### 4.3 Pourquoi cette approche est meilleure

En lisant la valeur depuis `config.py` plutôt qu'en la codant en dur, on garantit que :

- Si le modèle change (par exemple passage à MobileNetV2 qui accepte aussi 224x224), la configuration est centralisée en un seul endroit.
- Il est impossible d'avoir des incohérences entre différentes parties du code.
- La valeur est documentée et visible par tous les développeurs qui consultent `config.py`.

---

### 5. Vérification de la correction

Après application du correctif, les tests suivants ont été réalisés pour confirmer le bon fonctionnement :

#### 5.1 Test de l'endpoint `/health`

Requête :
```
GET http://localhost:8000/health
```

Réponse attendue et obtenue :
```json
{
  "statut": "ok",
  "modele": "ResNet50"
}
```

#### 5.2 Test de l'endpoint `/predict` avec une image de chien

Requête :
```
POST http://localhost:8000/predict
Content-Type: multipart/form-data
fichier: labrador.jpg
```

Réponse obtenue après correction :
```json
{
  "classe": "Labrador Retriever",
  "confiance_pct": 87.43,
  "top_5": "1. Labrador Retriever: 87.43%\n2. Golden Retriever: 6.21%\n3. ...",
  "temps_ms": 312
}
```

#### 5.3 Vérification des logs après correction

```
2025-04-28 10:52:11 — INFO — Prédiction OK : 'Labrador Retriever' | Confiance : 87.43% | Temps : 312ms
```

Plus aucune erreur dans `app.log`. Le compteur d'erreurs consécutives a été remis à zéro automatiquement.

#### 5.4 Vérification via l'endpoint `/metrics`

Requête :
```
GET http://localhost:8000/metrics
```

Réponse :
```json
{
  "predictions_totales": 1,
  "erreurs": 0,
  "confiance_faible": 0,
  "confiance_moyenne_pct": 87.43,
  "temps_reponse_moyen_ms": 312,
  "feedback_correct": 0,
  "feedback_incorrect": 0
}
```

---

### 6. Rôle du monitoring dans la détection de l'incident

Sans le système de monitoring et de journalisation mis en place (compétence C20), cet incident aurait été difficile à diagnostiquer rapidement. Voici ce que le monitoring a apporté :

- **Journalisation automatique** : le module `logging` a enregistré l'erreur précise avec son message technique dans `app.log`, permettant d'identifier immédiatement la cause.
- **Détection automatique d'incident** : après 3 erreurs consécutives, une alerte de niveau `CRITICAL` a été automatiquement générée, signalant qu'une intervention humaine était nécessaire.
- **Traçabilité** : chaque prédiction (réussie ou non) est horodatée dans `predictions_history.csv`, permettant de connaître exactement le moment et la fréquence des échecs.

Ce cas démontre concrètement l'intérêt d'une approche MLOps : la surveillance de l'application a permis de détecter, diagnostiquer et corriger l'incident en moins de 40 minutes.

---

### 7. Mesures préventives mises en place

Suite à cet incident, les mesures suivantes ont été adoptées pour éviter qu'un problème similaire se reproduise :

1. **Centralisation de tous les paramètres** dans `config.py` — plus aucune valeur numérique technique n'est codée en dur dans le code applicatif.

2. **Seuil d'alerte sur les erreurs consécutives** — configuré à 3 dans `MONITORING_CONFIG`, ce qui permet une détection rapide de tout dysfonctionnement.

3. **Endpoint `/health`** — permet de vérifier en un instant que l'API est opérationnelle avant de l'utiliser.

4. **Documentation systématique** dans ce fichier `INCIDENTS.md` — tout incident futur sera documenté selon le même modèle.

---

### 8. Résumé de l'incident

| Champ | Valeur |
|---|---|
| Identifiant | INCIDENT #001 |
| Date | 28 avril 2025 |
| Durée | 38 minutes |
| Gravité | Critique |
| Fichier concerné | `main.py` → `preprocess_image()` |
| Cause | Taille d'image incorrecte : `(64, 64)` au lieu de `(224, 224)` |
| Détection | Automatique via `logging` + alerte CRITICAL |
| Correction | Lecture de `MODEL_CONFIG["input_shape"]` depuis `config.py` |
| Statut | ✅ Résolu |

---

*Document rédigé dans le cadre de la compétence C21 — Résoudre les incidents techniques en apportant les modifications nécessaires au code de l'application et en documentant les solutions pour en garantir le fonctionnement opérationnel.*
