---
title: Classification d'Images IA
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: main.py
pinned: false
---

# 🖼️ Classification d'Images avec Intelligence Artificielle

Un système de classification d'images utilisant un modèle préentraîné ResNet50 pour identifier automatiquement le contenu des images.

## ✨ Fonctionnalités

- **Modèle préentraîné** : Utilise ResNet50 entraîné sur ImageNet
- **Interface web intuitive** : Interface Gradio avec glisser-déposer
- **Prédictions détaillées** : Affiche la classe principale et les 5 prédictions les plus probables
- **Sauvegarde automatique** : Toutes les prédictions sont sauvegardées dans un fichier CSV
- **Support multi-format** : Images locales et URLs supportées
- **Interface moderne** : Design épuré et responsive

## 🚀 Installation

### Prérequis
- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner ou télécharger le projet**
   ```bash
   git clone <url-du-repo>
   cd image_organisation
   ```

2. **Créer un environnement virtuel (recommandé)**
   ```bash
   python -m venv venv
   
   # Sur Windows
   venv\Scripts\activate
   
   # Sur macOS/Linux
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Utilisation

### Lancement de l'application

```bash
python main.py
```

L'interface web sera accessible à l'adresse : `http://localhost:7860`

### Comment utiliser l'application

1. **Charger une image** :
   - Glissez-déposez une image dans la zone de téléchargement
   - Ou cliquez sur la zone pour sélectionner un fichier
   - Ou utilisez les exemples fournis

2. **Analyser l'image** :
   - Cliquez sur le bouton "🔍 Analyser l'image"
   - Attendez quelques secondes pour le traitement

3. **Consulter les résultats** :
   - La classe prédite principale s'affiche
   - Le score de confiance est indiqué en pourcentage
   - Les 5 prédictions les plus probables sont listées

### Fichiers générés

- `predictions_history.csv` : Historique de toutes les prédictions effectuées

## 📊 Fonctionnalités techniques

### Modèle utilisé
- **ResNet50** préentraîné sur ImageNet
- Reconnaît plus de 1000 classes d'objets différents
- Précision élevée sur les objets courants

### Prétraitement des images
- Redimensionnement automatique à 224x224 pixels
- Normalisation selon les spécifications de ResNet50
- Support des formats : JPG, PNG, BMP, etc.

### Sauvegarde des données
- Timestamp de chaque prédiction
- Nom de l'image analysée
- Classe prédite et score de confiance
- Top 5 des prédictions complètes

## 🛠️ Structure du projet

```
image_organisation/
├── main.py                 # Application principale
├── requirements.txt        # Dépendances Python
├── README.md              # Documentation
└── predictions_history.csv # Historique des prédictions (généré automatiquement)
```

## 🔧 Personnalisation

### Modifier le modèle
Pour utiliser un autre modèle préentraîné, modifiez la ligne dans `main.py` :

```python
# Au lieu de ResNet50, vous pouvez utiliser :
from tensorflow.keras.applications import MobileNetV2, InceptionV3, VGG16

# Et changer la ligne d'initialisation :
self.model = MobileNetV2(weights='imagenet')  # ou autre modèle
```

### Ajouter de nouvelles fonctionnalités
- **Export des résultats** : Modifiez la fonction `save_prediction()`
- **Interface personnalisée** : Modifiez la fonction `create_interface()`
- **Prétraitement avancé** : Modifiez la fonction `preprocess_image()`

## 🐛 Dépannage

### Problèmes courants

1. **Erreur de mémoire** :
   - Fermez d'autres applications
   - Redémarrez l'application

2. **Téléchargement du modèle lent** :
   - Le modèle ResNet50 fait ~100MB
   - Le téléchargement se fait automatiquement au premier lancement

3. **Erreur de dépendances** :
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

### Logs et débogage
L'application affiche des messages de statut dans la console :
- Chargement du modèle
- Erreurs de prédiction
- Erreurs de sauvegarde

## 📈 Améliorations possibles

- [ ] Support de plusieurs modèles
- [ ] Interface pour l'entraînement personnalisé
- [ ] Export des résultats en différents formats
- [ ] Interface mobile responsive
- [ ] Intégration avec des APIs externes
- [ ] Support du traitement par lots

## 📝 Licence

Ce projet est créé pour un portfolio de démonstration. Libre d'utilisation et de modification.

## 🤝 Contribution

Les suggestions d'amélioration sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation

---

**Développé avec ❤️ pour démontrer les compétences en Python, IA et interfaces utilisateur** 

### Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.