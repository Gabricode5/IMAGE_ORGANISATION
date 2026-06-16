---
projet: IMAGE_ORGANISATION
auteur: Gabriel Guery
date: Mai 2026
version: "1.0"
stack: ResNet50 · FastAPI · Gradio
---

# Documentation Technique - Monitoring et Gestion des Incidents
## Projet IMAGE_ORGANISATION

---

## 1. Architecture du dispositif de monitoring

### 1.1 Composants du système

| Composant | Rôle | Technologie |
|---|---|---|
| Module de prédiction | Chargement du modèle ResNet50, prétraitement des images, inférence | TensorFlow / Keras |
| Fichier de configuration centralisé | Centralisation de tous les paramètres (modèle, seuils, RGPD) | Python dict |
| Système de journalisation | Enregistrement horodaté de chaque événement applicatif | `logging` Python |
| Historique des prédictions | Persistance des résultats, des scores de confiance et des retours utilisateurs | CSV (pandas) |
| Interface de monitoring | Tableau de bord temps réel des métriques clés | Gradio |
| API REST | Exposition des endpoints de prédiction, santé et métriques | FastAPI |

### 1.2 Flux de données - schéma ASCII

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUX D'EXÉCUTION                            │
└─────────────────────────────────────────────────────────────────────┘

  Requête utilisateur (image)
          │
          ▼
  ┌───────────────────┐
  │   Endpoint REST   │  POST /predict
  │     FastAPI       │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ preprocess_image()│  Redimensionnement → 224×224
  │  Module principal │  Normalisation ImageNet
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │    ResNet50       │  Inférence sur ImageNet (1 000 classes)
  │    TensorFlow     │
  └────────┬──────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
  Succès        Erreur
     │            │
     ▼            ▼
 INFO log     ERROR log ──→ compteur erreurs consécutives
     │            │                    │
     ▼            ▼               ≥ 3 erreurs
  Sauvegarde    Sauvegarde             │
  historique    historique             ▼
  (confiance,   (classe =          CRITICAL log
   temps, top5)  "Erreur…")        Alerte intervention
     │
     ▼
  Feedback utilisateur (Correcte / Incorrecte)
     │
     ▼
  Mise à jour historique ──→ Boucle d'amélioration continue
```

---

## 2. Conformité RGPD du monitoring

### 2.1 Données collectées vs données exclues

| Catégorie | Donnée | Collectée | Justification |
|---|---|---|---|
| Résultat d'inférence | Classe prédite (label) | Oui | Nécessaire au suivi qualité du modèle |
| Résultat d'inférence | Score de confiance (%) | Oui | Nécessaire à la détection d'anomalies |
| Performance | Temps de réponse (ms) | Oui | Nécessaire au monitoring de performance |
| Temporalité | Horodatage de la requête | Oui | Nécessaire à la traçabilité des incidents |
| Qualité | Top 5 des classes candidates | Oui | Nécessaire à l'analyse post-incident |
| Retour utilisateur | Feedback Correcte / Incorrecte | Oui | Alimentation volontaire de la boucle qualité |
| Données biométriques | Visages, identités, données sensibles | Non | Hors périmètre - aucune collecte |
| Données personnelles | Nom, email, adresse IP de l'utilisateur | Non | Non requises par le service |
| Contenu image | Fichier image source | Non | Aucun stockage - traitement en mémoire uniquement |

### 2.2 Politique de rétention

| Paramètre | Valeur configurée | Mécanisme appliqué |
|---|---|---|
| Volume maximal conservé | 1 000 entrées | Suppression automatique des entrées les plus anciennes au dépassement |
| Durée maximale de conservation | 90 jours | Purge automatique des entrées au-delà de 90 jours |
| Effacement à la demande | Disponible | Bouton dédié dans l'interface - réinitialisation immédiate de l'historique |
| Périmètre de stockage | Local uniquement | Aucune transmission vers un tiers ou un service cloud |
| Base légale du traitement | Intérêt légitime - amélioration d'un système IA | Limitation stricte à la finalité de monitoring |

---

## 3. Métriques surveillées et seuils d'alerte

| Métrique | Description | Unité | Seuil d'alerte | Niveau de sévérité | Action déclenchée |
|---|---|---|---|---|---|
| Score de confiance | Niveau de certitude du modèle sur la prédiction principale | % | < 30 % | WARNING | Journalisation de l'événement - signalement confiance faible |
| Erreurs consécutives | Nombre d'échecs de prédiction successifs sans succès intercalé | Nombre | ≥ 3 | CRITICAL | Alerte critique - intervention humaine requise |
| Taux d'erreur global | Ratio erreurs / prédictions totales | % | Indicatif | INFO | Affiché dans le tableau de bord - aucune action automatique |
| Temps de réponse moyen | Latence moyenne de traitement par requête | ms | Indicatif | INFO | Affiché dans le tableau de bord |
| Précision feedback | Ratio prédictions correctes / feedbacks soumis | % | Indicatif | INFO | Alimentation de la boucle d'amélioration continue |
| Volume historique | Nombre total d'entrées dans l'historique | Lignes | 1 000 | INFO | Troncature automatique - conformité RGPD |

---

## 4. Mécanismes de détection automatique

### 4.1 Détection des anomalies de qualité de prédiction

À chaque inférence réussie, le score de confiance de la classe principale est comparé au seuil configuré (30 %). Si ce seuil n'est pas atteint, un événement de niveau WARNING est automatiquement émis dans le fichier de logs, identifiant la classe prédite et le score obtenu. Ce mécanisme permet de détecter les images ambiguës, dégradées ou hors distribution d'entraînement sans interrompre le service.

```
Confiance obtenue < 30 %
        │
        ▼
WARNING - "Confiance faible : XX.XX% pour 'classe' (seuil : 30%)"
        │
        ▼
Prédiction sauvegardée dans l'historique (marquée comme suspecte par le score)
```

### 4.2 Détection des incidents de stabilité

Un compteur d'erreurs consécutives est maintenu en mémoire pendant l'exécution. À chaque exception levée lors d'une prédiction, ce compteur est incrémenté et un événement ERROR est journalisé. Dès que le compteur atteint ou dépasse le seuil de 3, un événement de niveau CRITICAL est émis, signalant qu'une intervention humaine est nécessaire. Le compteur est remis à zéro dès qu'une prédiction aboutit avec succès.

```
Exception lors de l'inférence
        │
        ▼
ERROR - message d'erreur technique
compteur_erreurs += 1
        │
   compteur ≥ 3 ?
   ┌────┴────┐
  Non       Oui
   │         │
  (fin)      ▼
         CRITICAL - "INCIDENT CRITIQUE - N erreurs consécutives ! Intervention requise."
```

### 4.3 Détection des dégradations de performance

À chaque prédiction, le temps de réponse mesuré est comparé au seuil de latence configuré (1 000 ms). Tout dépassement de ce seuil génère automatiquement un événement de niveau WARNING dans le fichier de logs, indiquant la valeur mesurée et le seuil de référence. Ce mécanisme permet de détecter une dégradation des performances avant qu'elle n'impacte l'expérience utilisateur. Le seuil est paramétrable dans le fichier de configuration centralisé.

```
Temps de réponse mesuré > 1 000 ms
        │
        ▼
WARNING - "Latence anormale : XXXXms (seuil : 1000ms)"
        │
        ▼
Événement journalisé - métriques de latence mises à jour dans l'historique
```

---

## 5. Fiche incident INC-001

### 5.1 Tableau synthétique

| Champ | Valeur |
|---|---|
| Identifiant | INC-001 |
| Date de détection | 28 avril 2025 - 10h14 |
| Date de résolution | 28 avril 2025 - 10h52 |
| Durée totale | 38 minutes |
| Gravité | Critique |
| Module impacté | Fonction de prétraitement des images - module de prédiction |
| Cause racine | Valeur de redimensionnement codée en dur incorrecte : `(64, 64)` au lieu de `(224, 224)` |
| Mode de détection | Automatique - journalisation + alerte CRITICAL après 3 erreurs consécutives |
| Périmètre d'impact | 100 % des requêtes - aucune prédiction possible |
| Statut | Résolu |

### 5.2 Symptôme - traces de journalisation

Toutes les requêtes envoyées à l'endpoint de prédiction retournaient une erreur HTTP 500. Le fichier de logs enregistrait automatiquement la séquence suivante :

```
2025-04-28 10:14:32 - INFO  - Initialisation du système de classification d'images...
2025-04-28 10:14:45 - INFO  - Modèle ResNet50 chargé avec succès.
2025-04-28 10:14:51 - INFO  - Fichier historique créé.
2025-04-28 10:15:03 - ERROR - Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:03 - ERROR - Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:03 - ERROR - Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
2025-04-28 10:15:04 - CRITICAL - INCIDENT CRITIQUE - 3 erreurs consécutives ! Intervention requise.
```

Le message d'erreur identifie précisément l'incompatibilité : le tenseur fourni au modèle présentait la forme `(None, 64, 64, 3)` alors que ResNet50 impose strictement `(None, 224, 224, 3)`.

### 5.3 Diagnostic - chaîne de traitement

L'analyse de la chaîne de traitement a permis d'isoler le point de défaillance dans la fonction de prétraitement des images :

```
Image reçue (format quelconque, dimensions quelconques)
        │
        ▼
preprocess_image()
  └─ img.resize((64, 64))   ◄── POINT DE DÉFAILLANCE
        │
        ▼
np.expand_dims() → tenseur shape (1, 64, 64, 3)
        │
        ▼
ResNet50.predict()
  └─ attend shape (1, 224, 224, 3)
        │
        ▼
EXCEPTION - incompatibilité de dimensions
        │
        ▼
ERROR log → CRITICAL log (après 3 occurrences)
```

**Cause racine identifiée :** la valeur `(64, 64)` avait été introduite en phase de développement pour accélérer les tests avec des images de petite taille. Elle n'avait pas été corrigée avant la mise en service, et aucun garde-fou (assertion, validation d'entrée) n'existait pour détecter cette incohérence à l'initialisation.

### 5.4 Correction - comparaison avant / après

**Avant correction** - valeur codée en dur, découplée du paramétrage central :

```python
# Valeur incorrecte - ne correspond pas aux contraintes de ResNet50
img = img.resize((64, 64))
```

**Après correction** - lecture depuis le fichier de configuration centralisé :

```python
# Valeur lue depuis la configuration centralisée - garantit la cohérence
img = img.resize(MODEL_CONFIG["input_shape"])  # → (224, 224)
```

Extrait correspondant du fichier de configuration centralisé :

```python
MODEL_CONFIG = {
    "model_name": "ResNet50",
    "weights": "imagenet",
    "input_shape": (224, 224),   # contrainte stricte de ResNet50
    "top_k": 5
}
```

### 5.5 Validation de la correction

| Test réalisé | Attendu | Obtenu | Résultat |
|---|---|---|---|
| Endpoint de santé `GET /health` | `{"statut": "ok", "modele": "ResNet50"}` | `{"statut": "ok", "modele": "ResNet50"}` | Succès |
| Prédiction image (labrador.jpg) | Classe identifiée, confiance > 30 % | `"Labrador Retriever"` - 87,43 % - 312 ms | Succès |
| Absence d'erreur dans les logs | Aucun événement ERROR ou CRITICAL | `INFO - Prédiction OK : 'Labrador Retriever' | Confiance : 87.43% | Temps : 312ms` | Succès |
| Remise à zéro du compteur d'erreurs | `consecutive_errors = 0` | Confirmé par endpoint métriques - `"erreurs": 0` | Succès |
| Endpoint métriques `GET /metrics` | Métriques cohérentes post-correction | `predictions_totales: 1, erreurs: 0, confiance_moyenne_pct: 87.43` | Succès |

### 5.6 Comparatif - impact du monitoring sur la résolution

| Dimension | Sans monitoring | Avec monitoring |
|---|---|---|
| Détection de l'incident | Signalement manuel par un utilisateur - délai indéterminé | Automatique - alerte CRITICAL après 3 erreurs consécutives (< 5 secondes) |
| Identification de la cause | Lecture du code source de bout en bout - investigation non guidée | Message d'erreur précis dans les logs : incompatibilité de forme `(64,64)` vs `(224,224)` |
| Délai de diagnostic | Estimé à plusieurs heures | Inférieur à 10 minutes - message d'erreur auto-documenté |
| Traçabilité | Aucune - perte de l'historique des échecs | Complète - horodatage, fréquence, nature de chaque erreur archivés |
| Délai total de résolution | Non estimable | 38 minutes - détection à correction incluse |

---

## 6. Mesures préventives et capitalisation

À la suite de l'incident INC-001, trois mesures structurantes ont été mises en place pour réduire la probabilité et l'impact de tout incident similaire :

**Mesure 1 - Centralisation de l'ensemble des paramètres techniques dans le fichier de configuration**

Aucune valeur numérique technique (taille d'image, seuil de confiance, limite de rétention) n'est désormais codée en dur dans le code applicatif. Toute modification de paramètre s'effectue en un point unique, éliminant les risques d'incohérence entre les différents modules de l'application - qu'il s'agisse du module de prédiction, de l'interface ou de l'API REST.

**Mesure 2 - Seuil d'alerte sur les erreurs consécutives configuré à 3 occurrences**

Le mécanisme de détection automatique d'incident est activé dès trois échecs successifs. Ce seuil, paramétrable dans le fichier de configuration centralisé, garantit qu'aucun dysfonctionnement systémique ne peut passer inaperçu plus de quelques secondes. Toute modification de seuil est traçable via le contrôle de version.

**Mesure 3 - Documentation systématique de chaque incident selon un modèle standardisé**

Chaque incident technique fait l'objet d'une fiche structurée couvrant : la chronologie, les traces de journalisation, le diagnostic de cause racine, la correction appliquée avec comparaison avant/après, et la validation. Cette capitalisation constitue une base de connaissance opérationnelle permettant de réduire le temps de résolution des incidents futurs.

**Mesure 4 - Endpoint de vérification préalable avec contrôles réels**

L'endpoint de santé vérifie trois conditions avant toute utilisation du service : le modèle est chargé en mémoire, le fichier d'historique est accessible en lecture/écriture, et le système de journalisation est disponible en écriture. Ce mécanisme permet de détecter une défaillance structurelle avant même la première prédiction, sans attendre qu'un utilisateur rencontre une erreur.

---

## 7. Boucle d'amélioration continue - feedback loop

### 7.1 Cycle d'amélioration - schéma ASCII

```
┌─────────────────────────────────────────────────────────────────────┐
│                  BOUCLE D'AMÉLIORATION CONTINUE                     │
└─────────────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────┐
         │    Prédiction produite          │
         │    (classe, confiance, top 5)   │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Évaluation par l'utilisateur  │
         │    ✅ Correcte / ❌ Incorrecte   │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Enregistrement du feedback   │
         │    dans l'historique persisté   │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Calcul de la précision       │
         │    (feedbacks corrects / total) │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Analyse des cas incorrects   │
         │    Identification des patterns  │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Décision d'amélioration      │
         │    (seuil, modèle, données)     │
         └────────────┬────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────┐
         │    Mise à jour de la config     │
         │    ou du modèle                 │
         └────────────┬────────────────────┘
                      │
                      └──────────► Nouvelle prédiction
```

### 7.2 Signaux de la boucle et décisions associées

| Signal | Source | Indicateur calculé | Décision possible |
|---|---|---|---|
| Feedback "Incorrecte" fréquent sur une catégorie | Retours utilisateurs | Taux d'erreur par classe (analyse manuelle du CSV) | Réévaluation du seuil de confiance - investigation des cas limites |
| Confiance moyenne en baisse | Métriques temps réel | `confiance_moyenne_pct` < valeur de référence | Vérification de la distribution des images soumises - suspicion de dérive |
| Augmentation du taux d'erreurs techniques | Logs d'erreur | `erreurs / predictions_totales` en hausse | Diagnostic infrastructure - analyse des exceptions journalisées |
| Score de confiance faible récurrent | Détection automatique WARNING | Nombre d'entrées sous le seuil de 30 % | Ajustement du seuil ou enrichissement des données d'entraînement |
| Alerte CRITICAL déclenchée | Détection automatique | Compteur erreurs consécutives ≥ 3 | Intervention humaine immédiate - diagnostic via les logs - correction du code |
| Volume historique proche de la limite RGPD | Contrôle de rétention | Nombre de lignes ≥ 1 000 | Troncature automatique - archivage optionnel avant suppression |

---

*Version 1.0 - Mai 2026 - Gabriel Guery - Projet IMAGE_ORGANISATION*
