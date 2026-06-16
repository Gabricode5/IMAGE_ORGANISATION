# Audit C20 — Système de monitoring de l'application IA

**Projet :** Classification d'Images IA (ResNet50 + FastAPI + Gradio)  
**Date de l'audit :** 09 mai 2026  
**Fichiers inspectés :** `main.py`, `api.py`, `config.py`, `requirements.txt`, `predictions_history.csv`, `INCIDENTS.md`

---

## Tableau récapitulatif

| # | Critère C20 | État | Détail |
|---|---|---|---|
| C20-1 | Métriques et seuils d'alerte documentés | ⚠️ Partiel | Seuils présents dans `config.py`, métriques calculées dans le code — aucun document de référence externe |
| C20-2 | Arguments en faveur des choix techniques | ❌ Manquant | Aucune justification rédigée des outils retenus |
| C20-3 | Outils installés et opérationnels | ⚠️ Partiel | `logging`, CSV, `/health`, `/metrics` opérationnels — rotation des logs absente |
| C20-4 | Règles de journalisation intégrées au code | ✅ Présent | `logging` Python configuré et utilisé dans tout `main.py` |
| C20-5 | Alertes configurées et opérationnelles | ✅ Présent | WARNING confiance faible + CRITICAL erreurs consécutives actifs |
| C20-6 | Procédure d'installation et de configuration | ❌ Manquant | Aucune procédure rédigée |
| C20-7 | Format accessible | ❌ Manquant | Aucun document `MONITORING.md` produit |

---

## Détail par élément inspecté

### 1. Système de journalisation

**État : ✅ Présent (rotation ❌ absente)**

| Élément | Valeur | Preuve |
|---|---|---|
| Module utilisé | `logging` (stdlib Python) | `main.py:11` |
| Configuration | `basicConfig` avec `LOGGING_CONFIG` | `main.py:18-25` |
| Format | `%(asctime)s — %(levelname)s — %(message)s` | `config.py:47` |
| Sortie fichier | `app.log` | `main.py:21` |
| Sortie console | `StreamHandler` | `main.py:22` |
| Rotation | **Absente** — `FileHandler` simple | `main.py:21` |

**Niveaux de log utilisés :**

| Niveau | Contexte | Fichier:ligne |
|---|---|---|
| INFO | Initialisation, chargement modèle, sauvegarde CSV | `main.py:31, 39, 47, 88-90, 127, 140` |
| WARNING | Confiance < 30%, image absente | `main.py:81-84, 174` |
| ERROR | Exception lors de prédiction ou sauvegarde | `main.py:98, 131, 144` |
| CRITICAL | ≥ 3 erreurs consécutives | `main.py:102-105` |

**Ce qui manque :** `RotatingFileHandler` — le fichier `app.log` croît indéfiniment sans limite de taille ni archivage.

---

### 2. Endpoints de monitoring

**État : ✅ Présents (health ⚠️ minimal)**

#### `/health` — `api.py:12-15`

```python
@app.get("/health")
def health():
    return {"statut": "ok", "modele": "ResNet50"}
```

- Public (pas d'authentification)
- **Problème :** retourne toujours `"ok"` sans vérifier si le modèle est réellement chargé ni si le fichier CSV est accessible. Réponse statique.

#### `/metrics` — `api.py:42-54`

```python
@app.get("/metrics")
def metrics():
    total, errors, low_conf, avg_conf, avg_time, correct_fb, incorrect_fb = classifier.get_metrics()
    return { ... }  # 7 métriques retournées
```

- Public
- Calcul dynamique depuis le CSV ✅
- **Ce qui manque :** la valeur courante de `consecutive_errors` (compteur en mémoire) n'est pas exposée — ni le taux d'erreur calculé.

---

### 3. Stockage historique des prédictions

**État : ✅ Présent**

| Élément | Valeur | Preuve |
|---|---|---|
| Fichier | `predictions_history.csv` | `config.py:27` |
| Création automatique | Oui, si absent au démarrage | `main.py:36-47` |
| Politique de rétention | 1 000 lignes max (LIFO) | `main.py:124-127`, `config.py:41` |
| Suppression manuelle | Oui, via bouton Gradio | `main.py:147-155` |

**Schéma CSV :**

| Colonne | Type | Description |
|---|---|---|
| `timestamp` | string | Horodatage de la prédiction |
| `image_name` | string | Nom générique (pas d'identification personnelle) |
| `predicted_class` | string | Classe retournée par ResNet50 |
| `confidence` | float | Score de confiance (%) |
| `top_5_predictions` | string | 5 meilleures classes avec scores |
| `response_time_ms` | float | Temps de traitement |
| `feedback` | string | "Correcte", "Incorrecte" ou vide |

**Conformité RGPD :**
- Aucune image stockée ✅
- Aucune adresse IP stockée ✅
- Aucune métadonnée EXIF ✅
- `image_name` = valeur générique `"image"` (API) ✅
- Suppression possible à tout moment ✅
- **Ce qui manque :** durée de conservation en jours non définie (seulement en nombre de lignes)

---

### 4. Détection automatique d'incidents

**État : ✅ Présent**

#### Confiance faible — `main.py:80-84`

```python
if main_confidence < MONITORING_CONFIG["confidence_threshold"]:  # seuil : 30%
    logger.warning(
        f"INCIDENT — Confiance faible : {main_confidence:.2f}% ..."
    )
```

- Seuil : **30%** (`config.py:39`)
- Niveau d'alerte : `WARNING`
- Déclenchement : à chaque prédiction

#### Erreurs consécutives — `main.py:97-105`

```python
self.consecutive_errors += 1
if self.consecutive_errors >= MONITORING_CONFIG["consecutive_errors_threshold"]:  # seuil : 3
    logger.critical("INCIDENT CRITIQUE — X erreurs consécutives !")
```

- Seuil : **3 erreurs consécutives** (`config.py:40`)
- Niveau d'alerte : `CRITICAL`
- Remise à zéro : à la première prédiction réussie (`main.py:92`)

**Ce qui manque :**
- Aucun seuil sur le **temps de réponse** (latence anormale non détectée)
- La valeur courante de `consecutive_errors` n'est pas exposée dans `/metrics`

---

### 5. Métriques exposées

**État : ✅ Présent**

Calculées dans `main.py:158-170`, exposées dans `api.py:42-54` :

| Métrique exposée | Calcul | Alerte associée |
|---|---|---|
| `predictions_totales` | `len(df)` | — |
| `erreurs` | lignes dont `predicted_class` commence par "Erreur" | — |
| `confiance_faible` | lignes dont `confidence < 30%` | WARNING dans logs |
| `confiance_moyenne_pct` | `df["confidence"].mean()` | — |
| `temps_reponse_moyen_ms` | `df["response_time_ms"].mean()` | — |
| `feedback_correct` | lignes `feedback == "Correcte"` | — |
| `feedback_incorrect` | lignes `feedback == "Incorrecte"` | — |

**Ce qui manque :**
- `taux_erreur_pct` (calculé dans l'interface Gradio mais non exposé dans l'API)
- `erreurs_consecutives_courantes` (valeur en mémoire, non exposée)

---

### 6. Configuration centralisée

**État : ✅ Présent**

`config.py` centralise tous les paramètres de monitoring :

| Paramètre | Valeur | Usage |
|---|---|---|
| `MONITORING_CONFIG["confidence_threshold"]` | `30.0` | Seuil WARNING confiance faible |
| `MONITORING_CONFIG["consecutive_errors_threshold"]` | `3` | Seuil CRITICAL erreurs consécutives |
| `MONITORING_CONFIG["max_history_rows"]` | `1000` | Rétention RGPD |
| `LOGGING_CONFIG["level"]` | `"INFO"` | Niveau minimal des logs |
| `LOGGING_CONFIG["format"]` | `"%(asctime)s — %(levelname)s — %(message)s"` | Format des logs |
| `FILE_CONFIG["log_file"]` | `"app.log"` | Fichier de log |
| `FILE_CONFIG["history_file"]` | `"predictions_history.csv"` | Fichier historique |

---

### 7. Conformité RGPD

**État : ✅ Présent (⚠️ durée temporelle manquante)**

| Point RGPD | État | Preuve |
|---|---|---|
| Images non stockées | ✅ | Aucune écriture binaire dans le code |
| Adresses IP non collectées | ✅ | Aucune capture de `request.client` |
| Métadonnées EXIF non extraites | ✅ | Aucun usage de `PIL.ExifTags` |
| Limitation du volume | ✅ | 1 000 lignes max, `main.py:124-127` |
| Suppression à la demande | ✅ | `clear_history()`, `main.py:147-155` |
| Log RGPD à la suppression | ✅ | `main.py:151` |
| Durée de conservation en jours | ❌ | Non définie — seulement en nombre de lignes |

---

## Récapitulatif des manques à corriger

### À implémenter dans le code (Phase 2)

| Priorité | Élément | Fichier cible | Impact C20 |
|---|---|---|---|
| 🔴 Haute | Rotation des logs (`RotatingFileHandler`) | `main.py` | C20-3, C20-4 |
| 🔴 Haute | Exposition de `consecutive_errors` dans `/metrics` | `api.py` + `main.py` | C20-1, C20-5 |
| 🟡 Moyenne | Taux d'erreur dans `/metrics` | `api.py` | C20-1 |
| 🟡 Moyenne | Amélioration de `/health` (vérifications réelles) | `api.py` | C20-3, C20-5 |
| 🟢 Basse | Seuil de latence anormale dans `config.py` + alerte WARNING | `config.py` + `main.py` | C20-1, C20-5 |

### À rédiger uniquement (Phase 3)

| Document | Critères couverts |
|---|---|
| `MONITORING.md` | C20-1, C20-2, C20-6, C20-7 |

---

*Audit produit avant toute modification du code — état du dépôt au commit `f6694ae`.*
