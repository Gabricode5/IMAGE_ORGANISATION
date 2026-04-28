"""
Configuration du projet de classification d'images
Ce fichier centralise tous les paramètres configurables du projet.
"""

# Configuration du modèle
MODEL_CONFIG = {
    "model_name": "ResNet50",  # ResNet50, MobileNetV2, InceptionV3, VGG16
    "weights": "imagenet",
    "input_shape": (224, 224),
    "top_k": 5
}

# Configuration de l'interface
INTERFACE_CONFIG = {
    "title": "Classification d'Images IA",
    "theme": "soft",
    "server_port": 7860,
    "server_name": "0.0.0.0",
    "share": True,
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

# Configuration du monitoring et de la détection d'incidents
MONITORING_CONFIG = {
    "confidence_threshold": 30.0,        # % en dessous duquel une prédiction est suspecte
    "consecutive_errors_threshold": 3,   # nb d'erreurs consécutives avant incident CRITIQUE
    "max_history_rows": 1000,            # RGPD : nombre max de lignes conservées dans le CSV
}

# Configuration des logs
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s — %(levelname)s — %(message)s",
    "file": "app.log"
}
