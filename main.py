import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import requests
from io import BytesIO

class ImageClassifier:
    def __init__(self):
        """Initialise le modèle de classification d'images"""
        # Chargement du modèle ResNet50 préentraîné sur ImageNet
        self.model = ResNet50(weights='imagenet')
        self.history_file = "predictions_history.csv"
        
        # Création du fichier CSV s'il n'existe pas
        if not os.path.exists(self.history_file):
            self._create_csv_file()
    
    def _create_csv_file(self):
        """Crée le fichier CSV pour sauvegarder l'historique des prédictions"""
        df = pd.DataFrame(columns=['timestamp', 'image_name', 'predicted_class', 'confidence', 'top_5_predictions'])
        df.to_csv(self.history_file, index=False)
    
    def preprocess_image(self, img):
        """Prétraite l'image pour le modèle ResNet50"""
        if isinstance(img, str):
            # Si c'est une URL
            if img.startswith('http'):
                response = requests.get(img)
                img = Image.open(BytesIO(response.content))
            else:
                # Si c'est un chemin local
                img = Image.open(img)
        
        # Redimensionner l'image à 224x224 (taille requise par ResNet50)
        img = img.resize((224, 224))
        
        # Convertir en array numpy
        img_array = image.img_to_array(img)
        
        # Ajouter la dimension du batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prétraiter selon les spécifications de ResNet50
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, img):
        """Effectue la prédiction sur l'image"""
        try:
            # Prétraitement de l'image
            processed_img = self.preprocess_image(img)
            
            # Prédiction
            predictions = self.model.predict(processed_img)
            
            # Décodage des prédictions (top 5)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Formatage des résultats
            results = []
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                confidence = float(score) * 100
                results.append(f"{i+1}. {label.replace('_', ' ').title()}: {confidence:.2f}%")
            
            # Classe principale et confiance
            main_class = decoded_predictions[0][1].replace('_', ' ').title()
            main_confidence = float(decoded_predictions[0][2]) * 100
            
            return main_class, main_confidence, "\n".join(results)
            
        except Exception as e:
            return f"Erreur: {str(e)}", 0.0, "Erreur lors de la prédiction"
    
    def save_prediction(self, image_name, predicted_class, confidence, top_5_predictions):
        """Sauvegarde la prédiction dans le fichier CSV"""
        try:
            df = pd.read_csv(self.history_file)
            new_row = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_name': image_name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_5_predictions': top_5_predictions
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.history_file, index=False)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
    
    def classify_image(self, input_image, image_name="image"):
        """Fonction principale pour la classification d'images"""
        if input_image is None:
            return "Aucune image fournie", 0.0, "Veuillez charger une image"
        
        # Effectuer la prédiction
        predicted_class, confidence, top_5_predictions = self.predict(input_image)
        
        # Sauvegarder la prédiction
        self.save_prediction(image_name, predicted_class, confidence, top_5_predictions)
        
        return predicted_class, confidence, top_5_predictions

def create_interface():
    """Crée une interface Gradio professionnelle et responsive pour la classification d'images."""
    classifier = ImageClassifier()

    custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
        color: #f5f6fa !important;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        position: relative;
        overflow: hidden;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: 1px;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
    }
    .tech-badge {
        display: inline-block;
        background: #ffb347;
        color: #232526;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.15rem;
        font-weight: 600;
    }
    .info-section {
        background: rgba(255,255,255,0.07);
        padding: 1.3rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(2px);
    }
    
    .stat-card {
        background: rgba(255,255,255,0.10);
        padding: 1.1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: transform 0.2s;
        min-width: 110px;
    }
    .stat-card:hover {
        transform: translateY(-2px) scale(1.03);
        background: rgba(118,75,162,0.18);
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffb347;
        margin-bottom: 0.3rem;
    }
    .stat-label {
        color: #e0e0e0;
        font-size: 0.95rem;
    }
    .feature-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.7rem;
        margin-top: 0.7rem;
    }
    .feature-item {
        display: flex;
        align-items: center;
        padding: 0.7rem;
        background: #232526;
        border-radius: 8px;
        transition: background 0.3s ease;
        color: #fff;
        font-size: 0.98rem;
    }
    .feature-item:hover {
        background: #764ba2;
    }
    .feature-icon {
        font-size: 1.3rem;
        margin-right: 0.7rem;
    }
    .gr-button, .analyze-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1.7rem !important;
        box-shadow: 0 2px 8px rgba(102,126,234,0.18);
        transition: background 0.2s, transform 0.2s;
    }
    .gr-button:hover, .analyze-btn:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%) !important;
        transform: scale(1.04);
    }
    .footer {
        text-align: center;
        color: #bdbdbd;
        margin-top: 2rem;
        font-size: 0.95rem;
        padding-bottom: 1rem;
    }
    /* Responsive mobile */
    @media (max-width: 700px) {
      .gradio-container {
        max-width: 100vw !important;
        padding: 0 !important;
      }
      .main-header, .info-section {
        padding: 0.7rem !important;
        border-radius: 7px !important;
      }
      .main-header h1 {
        font-size: 1.2rem !important;
      }
      .feature-list {
        grid-template-columns: 1fr !important;
      }
      .stat-card {
        min-width: 80px !important;
        font-size: 0.85rem !important;
        padding: 0.7rem !important;
      }
    }
    """

    with gr.Blocks(title="Classification d'Images IA", theme=gr.themes.Soft(), css=custom_css) as interface:
        # En-tête principal
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 Classification d'Images IA</h1>
            <p>Reconnaissance d'objets par intelligence artificielle</p>
            <div style="margin-top: 0.7rem;">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">TensorFlow</span>
                <span class="tech-badge">ResNet50</span>
                <span class="tech-badge">Gradio</span>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Zone de Téléchargement")
                input_image = gr.Image(
                    label="Glissez-déposez votre image ici ou cliquez pour sélectionner",
                    type="pil",
                    height=320,
                    container=True
                )
                classify_btn = gr.Button(
                    "🔍 Analyser l'Image",
                    variant="primary",
                    size="lg",
                    elem_classes=["analyze-btn"]
                )
                gr.HTML('</div>')
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Résultats de l'Analyse")
                predicted_class = gr.Textbox(
                    label="🏷️ Classe Prédite",
                    interactive=False,
                    container=True
                )
                confidence = gr.Number(
                    label="📈 Niveau de Confiance (%)",
                    interactive=False,
                    container=True
                )
                top_5_predictions = gr.Textbox(
                    label="🏆 Top 5 des Prédictions",
                    interactive=False,
                    lines=8,
                    container=True
                )
                gr.HTML('</div>')

        # Historique des prédictions
        gr.HTML("""
        <div class="info-section">
            <h3>🕑 Historique des Prédictions</h3>
        </div>
        """)
        history_df = gr.Dataframe(
            headers=["Horodatage", "Nom de l'image", "Classe prédite", "Confiance", "Top 5"],
            datatype=["str", "str", "str", "number", "str"],
            interactive=False,
            row_count=5,
            label="Historique des prédictions récentes"
        )

        # Statistiques
        gr.HTML("""
        <div class="info-section">
            <h3>📈 Statistiques du Système</h3>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <div class="stat-card">
                    <div class="stat-number">1000+</div>
                    <div class="stat-label">Classes reconnues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">2-3s</div>
                    <div class="stat-label">Temps de traitement</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Précision ImageNet</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Disponibilité</div>
                </div>
            </div>
        </div>
        """)

        # Fonctionnalités
        gr.HTML("""
        <div class="info-section">
            <h3>✨ Fonctionnalités Avancées</h3>
            <div class="feature-list">
                <div class="feature-item">
                    <span class="feature-icon">🧠</span>
                    <div>
                        <strong>Modèle Préentraîné</strong><br>
                        ResNet50 optimisé sur ImageNet
                    </div>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    <div>
                        <strong>Prédictions Détaillées</strong><br>
                        Top 5 avec scores de confiance
                    </div>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">💾</span>
                    <div>
                        <strong>Sauvegarde Automatique</strong><br>
                        Historique CSV complet
                    </div>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🌐</span>
                    <div>
                        <strong>Interface Web Responsive</strong><br>
                        Design moderne et adaptatif
                    </div>
                </div>
            </div>
        </div>
        """)

        # Guide d'utilisation
        gr.HTML("""
        <div class="info-section">
            <h3>📋 Guide d'Utilisation</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem;">
                <div>
                    <h4>🎯 Comment Utiliser</h4>
                    <ol style="margin-left: 1rem;">
                        <li><strong>Glissez-déposez</strong> une image dans la zone</li>
                        <li><strong>Cliquez</strong> sur "Analyser l'Image"</li>
                        <li><strong>Attendez</strong> 2-3 secondes</li>
                        <li><strong>Consultez</strong> les résultats détaillés</li>
                    </ol>
                </div>
                <div>
                    <h4>💡 Conseils d'Optimisation</h4>
                    <ul style="margin-left: 1rem;">
                        <li>Utilisez des images claires et bien éclairées</li>
                        <li>Évitez les images trop petites ou floues</li>
                        <li>Le modèle reconnaît 1000+ types d'objets</li>
                        <li>Toutes les prédictions sont sauvegardées</li>
                    </ul>
                </div>
            </div>
        </div>
        """)

        # Footer
        gr.HTML("""
        <div class="footer">
            © 2025 Gabri — Projet IA ResNet50 • Design par <a href="https://github.com/Gabricode5" style="color:#ffb347;text-decoration:none;">GitHub</a>
        </div>
        """)

        # Fonction pour classification + mise à jour historique
        def classify_and_update(image):
            result = classifier.classify_image(image)
            try:
                df = pd.read_csv(classifier.history_file)
                df = df.tail(10)
            except Exception:
                df = pd.DataFrame(columns=['timestamp', 'image_name', 'predicted_class', 'confidence', 'top_5_predictions'])
            return (*result, df)

        classify_btn.click(
            fn=classify_and_update,
            inputs=[input_image],
            outputs=[predicted_class, confidence, top_5_predictions, history_df]
        )

        # Affichage initial de l'historique au chargement
        def load_history():
            try:
                df = pd.read_csv(classifier.history_file)
                df = df.tail(10)
            except Exception:
                df = pd.DataFrame(columns=['timestamp', 'image_name', 'predicted_class', 'confidence', 'top_5_predictions'])
            return df

        interface.load(load_history, inputs=None, outputs=history_df)

    return interface

if __name__ == "__main__":
    print("🚀 Démarrage du système de classification d'images...")
    print("📦 Chargement du modèle ResNet50...")
    
    # Création et lancement de l'interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Active le partage public
        show_error=True
    ) 