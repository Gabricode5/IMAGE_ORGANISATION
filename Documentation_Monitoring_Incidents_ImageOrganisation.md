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

### 1.2 Flux de données

À chaque requête, l'image est prétraitée, soumise au modèle, puis le résultat est journalisé, persisté dans l'historique et vérifié contre les seuils d'alerte configurés.

---

## 2. Conformité RGPD du monitoring

### 2.1 Données collectées et politique de rétention

| Donnée | Collectée |
|---|---|
| Classe prédite (label) | Oui |
| Score de confiance (%) | Oui |
| Temps de réponse (ms) | Oui |
| Horodatage de la requête | Oui |
| Top 5 des classes candidates | Oui |
| Feedback Correcte / Incorrecte | Oui |
| Visages, identités, données sensibles | Non |
| Nom, email, adresse IP de l'utilisateur | Non |
| Fichier image source | Non |

L'historique est limité à 1 000 entrées et 90 jours (purge automatique). Un bouton dédié dans l'interface permet l'effacement immédiat à la demande. Aucune donnée n'est transmise vers un tiers ou un service cloud — stockage local uniquement, base légale : intérêt légitime (amélioration d'un système IA).

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

### 4.2 Détection des incidents de stabilité

Un compteur d'erreurs consécutives est maintenu en mémoire pendant l'exécution. À chaque exception levée lors d'une prédiction, ce compteur est incrémenté et un événement ERROR est journalisé. Dès que le compteur atteint ou dépasse le seuil de 3, un événement de niveau CRITICAL est émis, signalant qu'une intervention humaine est nécessaire. Le compteur est remis à zéro dès qu'une prédiction aboutit avec succès.

### 4.3 Détection des dégradations de performance

À chaque prédiction, le temps de réponse mesuré est comparé au seuil de latence configuré (1 000 ms). Tout dépassement de ce seuil génère automatiquement un événement de niveau WARNING dans le fichier de logs, indiquant la valeur mesurée et le seuil de référence. Ce mécanisme permet de détecter une dégradation des performances avant qu'elle n'impacte l'expérience utilisateur. Le seuil est paramétrable dans le fichier de configuration centralisé.

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
2025-04-28 10:15:03 - ERROR - Erreur lors de la prédiction :
    Input 0 of layer "resnet50" is incompatible with the layer:
    expected shape=(None, 224, 224, 3), found shape=(None, 64, 64, 3)
[×3 occurrences]
2025-04-28 10:15:04 - CRITICAL - INCIDENT CRITIQUE - 3 erreurs consécutives ! Intervention requise.
```

Le message d'erreur identifie précisément l'incompatibilité : le tenseur fourni au modèle présentait la forme `(None, 64, 64, 3)` alors que ResNet50 impose strictement `(None, 224, 224, 3)`.

### 5.3 Diagnostic - cause racine

**Cause racine identifiée :** la valeur `(64, 64)` avait été introduite en phase de développement pour accélérer les tests avec des images de petite taille. Elle n'avait pas été corrigée avant la mise en service, et aucun garde-fou (assertion, validation d'entrée) n'existait pour détecter cette incohérence à l'initialisation.

### 5.4 Correction - comparaison avant / après

**Avant correction** - valeur codée en dur, découplée du paramétrage central :

```python
img = img.resize((64, 64))
```

**Après correction** - lecture depuis le fichier de configuration centralisé :

```python
img = img.resize(MODEL_CONFIG["input_shape"])  # → (224, 224)
```

### 5.5 Validation de la correction

| Test réalisé | Attendu | Obtenu | Résultat |
|---|---|---|---|
| Prédiction image (labrador.jpg) | Classe identifiée, confiance > 30 % | `"Labrador Retriever"` - 87,43 % - 312 ms | Succès |
| Absence d'erreur dans les logs | Aucun événement ERROR ou CRITICAL | `INFO - Prédiction OK : 'Labrador Retriever' | Confiance : 87.43% | Temps : 312ms` | Succès |
| Endpoint métriques `GET /metrics` | Métriques cohérentes post-correction | `predictions_totales: 1, erreurs: 0, confiance_moyenne_pct: 87.43` | Succès |

### 5.6 Comparatif - impact du monitoring sur la résolution

| Dimension | Sans monitoring | Avec monitoring |
|---|---|---|
| Détection de l'incident | Signalement manuel par un utilisateur - délai indéterminé | Automatique - alerte CRITICAL après 3 erreurs consécutives (< 5 secondes) |
| Identification de la cause | Lecture du code source de bout en bout - investigation non guidée | Message d'erreur précis dans les logs : incompatibilité de forme `(64,64)` vs `(224,224)` |
| Délai total de résolution | Non estimable | 38 minutes - détection à correction incluse |

---

## 6. Mesures préventives et capitalisation

À la suite de l'incident INC-001, quatre mesures structurantes ont été mises en place pour réduire la probabilité et l'impact de tout incident similaire :

**Mesure 1 - Centralisation des paramètres** : plus aucune valeur technique codée en dur — toute modification s'effectue dans le fichier de configuration centralisé.

**Mesure 2 - Seuil d'alerte à 3 erreurs consécutives** : détection automatique d'incident dès trois échecs successifs, seuil paramétrable.

**Mesure 3 - Documentation standardisée des incidents** : chaque incident fait l'objet d'une fiche structurée (chronologie, diagnostic, correction, validation).

**Mesure 4 - Endpoint de santé avec contrôles réels** : vérification du modèle en mémoire, de l'historique et du système de journalisation avant la première prédiction.

---

## 7. Bilan

Le dispositif de monitoring couvre trois axes de surveillance : la qualité des prédictions (seuil de confiance < 30 %), la stabilité du service (alerte CRITICAL après 3 erreurs consécutives) et les performances (latence > 1 000 ms). L'ensemble fonctionne sans infrastructure externe, avec un coût de maintenance quasi nul.

L'incident INC-001 a validé l'efficacité du dispositif : détection automatique en moins de 5 secondes, diagnostic immédiat via les logs, résolution complète en 38 minutes.

Les mesures préventives adoptées (centralisation des paramètres, alertes automatiques, endpoint de santé, documentation systématique) réduisent le risque de récurrence et accélèrent la résolution de tout incident futur.

Les données collectées alimentent une boucle d'amélioration continue : les prédictions à faible confiance et les feedbacks négatifs identifient les faiblesses du modèle en production, orientant les décisions d'ajustement des seuils, d'enrichissement des données d'entraînement ou de remplacement du modèle. Ce cycle (prédiction → feedback → analyse → amélioration → redéploiement) est opérationnel dès le premier lancement.
