"""
Configuration du projet de classification d'images
Ce fichier centralise tous les paramètres configurables du projet.
"""

# Configuration du modèle
MODEL_CONFIG = {
    "model_name": "ResNet50",  # ResNet50, MobileNetV2, InceptionV3, VGG16
    "weights": "imagenet",
    "input_shape": (224, 224),  # Taille d'entrée du modèle
    "top_k": 5  # Nombre de prédictions à afficher
}

# Configuration de l'interface
INTERFACE_CONFIG = {
    "title": "Classification d'Images IA",
    "theme": "soft",  # soft, default, glass, monochrome
    "server_port": 2222,
    "server_name": "sftp://etu.bts-malraux72.net",
    "share": False,
    "show_error": True
}

# Configuration des fichiers
FILE_CONFIG = {
    "history_file": "predictions_history.csv",
    "log_file": "app.log"
}

# Configuration du prétraitement
PREPROCESSING_CONFIG = {
    "target_size": (224, 224),
    "color_mode": "rgb",
    "interpolation": "bilinear"
}

# Exemples d'images pour la démonstration
DEMO_IMAGES = [
    {
        "name": "Labrador",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/1200px-Labrador_on_Quantock_%282175262184%29.jpg",
        "description": "Chien Labrador"
    },
    {
        "name": "Voiture de sport",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Red_sports_car.jpg/1200px-Red_sports_car.jpg",
        "description": "Voiture de sport rouge"
    },
    {
        "name": "Oiseau (Martin-pêcheur)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Common_kingfisher_%28Alcedo_atthis%29.jpg/1200px-Common_kingfisher_%28Alcedo_atthis%29.jpg",
        "description": "Martin-pêcheur d'Europe"
    }
]

# Messages de l'interface
MESSAGES = {
    "welcome": """
    # 🖼️ Classification d'Images avec Intelligence Artificielle
    
    **Chargez une image** et découvrez ce que l'IA pense voir !
    
    Ce système utilise le modèle **ResNet50** préentraîné sur ImageNet pour classifier vos images.
    """,
    
    "instructions": """
    ### 📋 Instructions
    1. **Glissez-déposez** une image ou **cliquez** pour sélectionner un fichier
    2. Cliquez sur **"Analyser l'image"** pour obtenir la prédiction
    3. Les résultats incluent la classe principale et les 5 prédictions les plus probables
    4. Toutes les prédictions sont automatiquement sauvegardées dans `predictions_history.csv`
    
    ### 💡 Conseils
    - Utilisez des images claires et bien éclairées pour de meilleurs résultats
    - Le modèle reconnaît plus de 1000 classes d'objets différents
    - Les prédictions sont basées sur le dataset ImageNet
    """,
    
    "error_no_image": "Aucune image fournie",
    "error_processing": "Erreur lors de la prédiction",
    "error_save": "Erreur lors de la sauvegarde"
}

# Configuration des logs
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "app.log"
} 