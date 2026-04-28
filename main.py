import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
import logging
from PIL import Image
import requests
from io import BytesIO
from config import MODEL_CONFIG, INTERFACE_CONFIG, FILE_CONFIG, MONITORING_CONFIG, LOGGING_CONFIG

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"], encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ImageClassifier")


class ImageClassifier:
    def __init__(self):
        logger.info("Initialisation du système de classification d'images...")
        self.model = ResNet50(weights=MODEL_CONFIG["weights"])
        self.history_file = FILE_CONFIG["history_file"]
        self.consecutive_errors = 0

        if not os.path.exists(self.history_file):
            self._create_csv_file()

        logger.info(f"Modèle {MODEL_CONFIG['model_name']} chargé avec succès.")

    def _create_csv_file(self):
        df = pd.DataFrame(columns=[
            "timestamp", "image_name", "predicted_class",
            "confidence", "top_5_predictions", "response_time_ms", "feedback"
        ])
        df.to_csv(self.history_file, index=False)
        logger.info(f"Fichier historique créé : {self.history_file}")

    def preprocess_image(self, img):
        if isinstance(img, str):
            if img.startswith("http"):
                response = requests.get(img, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(img)

        img = img.resize(MODEL_CONFIG["input_shape"])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, img):
        start_time = time.time()
        try:
            processed_img = self.preprocess_image(img)
            predictions = self.model.predict(processed_img, verbose=0)
            decoded = decode_predictions(predictions, top=MODEL_CONFIG["top_k"])[0]
            response_time_ms = (time.time() - start_time) * 1000

            results = []
            for i, (_, label, score) in enumerate(decoded):
                confidence = float(score) * 100
                results.append(f"{i+1}. {label.replace('_', ' ').title()}: {confidence:.2f}%")

            main_class = decoded[0][1].replace("_", " ").title()
            main_confidence = float(decoded[0][2]) * 100

            # ── Détection d'incident : confiance faible ───────────────────
            if main_confidence < MONITORING_CONFIG["confidence_threshold"]:
                logger.warning(
                    f"INCIDENT — Confiance faible : {main_confidence:.2f}% "
                    f"pour '{main_class}' (seuil : {MONITORING_CONFIG['confidence_threshold']}%)"
                )
            else:
                logger.info(
                    f"Prédiction OK : '{main_class}' | "
                    f"Confiance : {main_confidence:.2f}% | "
                    f"Temps : {response_time_ms:.0f}ms"
                )

            self.consecutive_errors = 0
            return main_class, main_confidence, "\n".join(results), response_time_ms

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.consecutive_errors += 1
            logger.error(f"Erreur lors de la prédiction : {e}")

            # ── Détection d'incident : erreurs consécutives ───────────────
            if self.consecutive_errors >= MONITORING_CONFIG["consecutive_errors_threshold"]:
                logger.critical(
                    f"INCIDENT CRITIQUE — {self.consecutive_errors} erreurs consécutives ! "
                    "Intervention requise."
                )

            return f"Erreur: {str(e)}", 0.0, "Erreur lors de la prédiction", response_time_ms

    def save_prediction(self, image_name, predicted_class, confidence, top_5, response_time_ms):
        try:
            df = pd.read_csv(self.history_file)
            new_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": image_name,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "top_5_predictions": top_5,
                "response_time_ms": round(response_time_ms, 0),
                "feedback": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # RGPD : limite de rétention des données
            max_rows = MONITORING_CONFIG["max_history_rows"]
            if len(df) > max_rows:
                df = df.tail(max_rows)
                logger.info(f"RGPD : historique limité à {max_rows} entrées (anciennes données supprimées)")

            df.to_csv(self.history_file, index=False)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}")

    # ── Feedback loop ─────────────────────────────────────────────────────────
    def record_feedback(self, feedback_value):
        try:
            df = pd.read_csv(self.history_file)
            if len(df) > 0:
                df.loc[df.index[-1], "feedback"] = feedback_value
                df.to_csv(self.history_file, index=False)
                logger.info(f"Feedback loop alimentée : '{feedback_value}' sur la dernière prédiction")
                return f"✅ Feedback '{feedback_value}' enregistré."
            return "Aucune prédiction à évaluer."
        except Exception as e:
            logger.error(f"Erreur enregistrement feedback : {e}")
            return "❌ Erreur lors de l'enregistrement."

    # ── RGPD : effacement des données ─────────────────────────────────────────
    def clear_history(self):
        try:
            self._create_csv_file()
            logger.info("RGPD : historique effacé à la demande de l'utilisateur")
            return "✅ Historique effacé avec succès."
        except Exception as e:
            logger.error(f"Erreur effacement historique : {e}")
            return "❌ Erreur lors de l'effacement."

    # ── Métriques de monitoring ───────────────────────────────────────────────
    def get_metrics(self):
        try:
            df = pd.read_csv(self.history_file)
            total = len(df)
            errors = len(df[df["predicted_class"].str.startswith("Erreur", na=False)])
            low_conf = len(df[df["confidence"] < MONITORING_CONFIG["confidence_threshold"]])
            avg_conf = round(df["confidence"].mean(), 1) if total > 0 else 0.0
            avg_time = round(df["response_time_ms"].mean(), 0) if total > 0 else 0.0
            correct_fb = len(df[df["feedback"] == "Correcte"])
            incorrect_fb = len(df[df["feedback"] == "Incorrecte"])
            return total, errors, low_conf, avg_conf, avg_time, correct_fb, incorrect_fb
        except Exception:
            return 0, 0, 0, 0.0, 0.0, 0, 0

    def classify_image(self, input_image, image_name="image"):
        if input_image is None:
            logger.warning("Tentative de classification sans image fournie.")
            return "Aucune image fournie", 0.0, "Veuillez charger une image", 0.0

        predicted_class, confidence, top_5, response_time_ms = self.predict(input_image)
        self.save_prediction(image_name, predicted_class, confidence, top_5, response_time_ms)
        return predicted_class, confidence, top_5, response_time_ms


def create_interface():
    classifier = ImageClassifier()

    # ── Génération du HTML de monitoring ─────────────────────────────────────
    def generate_metrics_html():
        total, errors, low_conf, avg_conf, avg_time, correct_fb, incorrect_fb = classifier.get_metrics()
        error_rate = f"{(errors / total * 100):.1f}%" if total > 0 else "0%"
        fb_total = correct_fb + incorrect_fb
        accuracy = f"{(correct_fb / fb_total * 100):.1f}%" if fb_total > 0 else "N/A"
        error_color = "#ff4d4d" if errors > 0 else "#4dff91"
        low_conf_color = "#ffaa00" if low_conf > 0 else "#4dff91"
        return f"""
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;">
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Prédictions totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:{error_color}">{errors}</div>
                <div class="stat-label">Erreurs ({error_rate})</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:{low_conf_color}">{low_conf}</div>
                <div class="stat-label">Confiance faible</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_conf}%</div>
                <div class="stat-label">Confiance moyenne</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_time}ms</div>
                <div class="stat-label">Temps de réponse moyen</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{accuracy}</div>
                <div class="stat-label">Précision (feedback)</div>
            </div>
        </div>
        """

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
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: 1px; }
    .main-header p  { font-size: 1.1rem; opacity: 0.95; margin: 0; }
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
    .stat-card:hover { transform: translateY(-2px) scale(1.03); background: rgba(118,75,162,0.18); }
    .stat-number { font-size: 1.5rem; font-weight: bold; color: #ffb347; margin-bottom: 0.3rem; }
    .stat-label  { color: #e0e0e0; font-size: 0.95rem; }
    .feature-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.7rem; margin-top: 0.7rem; }
    .feature-item {
        display: flex; align-items: center; padding: 0.7rem;
        background: #232526; border-radius: 8px; transition: background 0.3s ease;
        color: #fff; font-size: 0.98rem;
    }
    .feature-item:hover { background: #764ba2; }
    .feature-icon { font-size: 1.3rem; margin-right: 0.7rem; }
    .gr-button, .analyze-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important; font-weight: 600 !important; border-radius: 8px !important;
        font-size: 1.05rem !important; padding: 0.7rem 1.7rem !important;
        box-shadow: 0 2px 8px rgba(102,126,234,0.18); transition: background 0.2s, transform 0.2s;
    }
    .gr-button:hover, .analyze-btn:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%) !important; transform: scale(1.04);
    }
    .footer { text-align: center; color: #bdbdbd; margin-top: 2rem; font-size: 0.95rem; padding-bottom: 1rem; }
    @media (max-width: 700px) {
        .gradio-container { max-width: 100vw !important; padding: 0 !important; }
        .main-header, .info-section { padding: 0.7rem !important; border-radius: 7px !important; }
        .main-header h1 { font-size: 1.2rem !important; }
        .feature-list { grid-template-columns: 1fr !important; }
        .stat-card { min-width: 80px !important; font-size: 0.85rem !important; padding: 0.7rem !important; }
    }
    """

    with gr.Blocks(title="Classification d'Images IA", theme=gr.themes.Soft(), css=custom_css) as interface:

        # ── En-tête ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 Classification d'Images IA</h1>
            <p>Reconnaissance d'objets par intelligence artificielle</p>
            <div style="margin-top:0.7rem;">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">TensorFlow</span>
                <span class="tech-badge">ResNet50</span>
                <span class="tech-badge">Gradio</span>
                <span class="tech-badge">MLOps</span>
            </div>
        </div>
        """)

        # ── Zone principale ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Zone de Téléchargement")
                input_image = gr.Image(
                    label="Glissez-déposez votre image ici ou cliquez pour sélectionner",
                    type="pil", height=320, container=True
                )
                classify_btn = gr.Button(
                    "🔍 Analyser l'Image", variant="primary", size="lg",
                    elem_classes=["analyze-btn"]
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Résultats de l'Analyse")
                predicted_class = gr.Textbox(label="🏷️ Classe Prédite", interactive=False)
                confidence = gr.Number(label="📈 Niveau de Confiance (%)", interactive=False)
                response_time = gr.Number(label="⏱️ Temps de Réponse (ms)", interactive=False)
                top_5_predictions = gr.Textbox(
                    label="🏆 Top 5 des Prédictions", interactive=False, lines=7
                )

                # ── Feedback loop ─────────────────────────────────────────────
                gr.Markdown("### 💬 Évaluation — Feedback Loop")
                gr.HTML("<p style='color:#bdbdbd;font-size:0.92rem;margin:0 0 0.5rem;'>La prédiction était-elle correcte ? Votre retour alimente la boucle d'amélioration continue.</p>")
                with gr.Row():
                    btn_correct = gr.Button("✅ Correcte", variant="secondary", size="sm")
                    btn_incorrect = gr.Button("❌ Incorrecte", variant="secondary", size="sm")
                feedback_status = gr.Textbox(label="Statut du feedback", interactive=False)

        # ── Monitoring en temps réel ──────────────────────────────────────────
        gr.HTML("""
        <div class="info-section">
            <h3>📡 Monitoring en Temps Réel</h3>
            <p style="color:#bdbdbd;font-size:0.9rem;margin:0 0 0.5rem;">
                Métriques calculées à partir de l'historique des prédictions.
                Les incidents (confiance &lt; 30%, erreurs consécutives) sont automatiquement
                journalisés dans <code>app.log</code>.
            </p>
        </div>
        """)
        metrics_html = gr.HTML()
        refresh_btn = gr.Button("🔄 Actualiser les métriques", size="sm")

        # ── Historique ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="info-section" style="margin-top:1rem;">
            <h3>🕑 Historique des Prédictions</h3>
        </div>
        """)
        history_df = gr.Dataframe(interactive=False, row_count=10)

        # ── RGPD ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="info-section">
            <h3>🔒 Gestion des Données Personnelles — RGPD</h3>
            <p style="color:#bdbdbd;font-size:0.92rem;">
                Les données collectées (horodatage, classe prédite, confiance) sont conservées
                localement dans <code>predictions_history.csv</code> avec une limite de
                <strong>1 000 entrées</strong>. Aucune image n'est stockée.
                Vous pouvez effacer l'intégralité de l'historique à tout moment.
            </p>
        </div>
        """)
        with gr.Row():
            clear_btn = gr.Button("🗑️ Effacer l'historique", variant="stop", size="sm")
            clear_status = gr.Textbox(label="Statut", interactive=False, scale=3)

        # ── Statistiques système ──────────────────────────────────────────────
        gr.HTML("""
        <div class="info-section">
            <h3>📈 Statistiques du Système</h3>
            <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                <div class="stat-card"><div class="stat-number">1000+</div><div class="stat-label">Classes reconnues</div></div>
                <div class="stat-card"><div class="stat-number">2-3s</div><div class="stat-label">Temps de traitement</div></div>
                <div class="stat-card"><div class="stat-number">95%</div><div class="stat-label">Précision ImageNet</div></div>
                <div class="stat-card"><div class="stat-number">24/7</div><div class="stat-label">Disponibilité</div></div>
            </div>
        </div>
        """)

        # ── Fonctionnalités ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="info-section">
            <h3>✨ Fonctionnalités Avancées</h3>
            <div class="feature-list">
                <div class="feature-item"><span class="feature-icon">🧠</span><div><strong>Modèle Préentraîné</strong><br>ResNet50 optimisé sur ImageNet</div></div>
                <div class="feature-item"><span class="feature-icon">📡</span><div><strong>Monitoring Temps Réel</strong><br>Métriques et détection d'incidents</div></div>
                <div class="feature-item"><span class="feature-icon">📋</span><div><strong>Journalisation complète</strong><br>Logs horodatés dans app.log</div></div>
                <div class="feature-item"><span class="feature-icon">🔁</span><div><strong>Feedback Loop MLOps</strong><br>Évaluation des prédictions</div></div>
                <div class="feature-item"><span class="feature-icon">🔒</span><div><strong>Conformité RGPD</strong><br>Rétention limitée, effacement possible</div></div>
                <div class="feature-item"><span class="feature-icon">💾</span><div><strong>Historique CSV</strong><br>Sauvegarde automatique complète</div></div>
            </div>
        </div>
        """)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
            © 2025 Gabri — Projet IA ResNet50 • MLOps & Monitoring •
            <a href="https://github.com/Gabricode5" style="color:#ffb347;text-decoration:none;">GitHub</a>
        </div>
        """)

        # ── Handlers ──────────────────────────────────────────────────────────
        def classify_and_update(img):
            predicted, conf, top5, resp_time = classifier.classify_image(img)
            try:
                df = pd.read_csv(classifier.history_file).tail(10)
                df = df.rename(columns={
                    "timestamp": "Horodatage", "image_name": "Image",
                    "predicted_class": "Classe prédite", "confidence": "Confiance (%)",
                    "top_5_predictions": "Top 5", "response_time_ms": "Temps (ms)",
                    "feedback": "Feedback"
                })
            except Exception:
                df = pd.DataFrame()
            return predicted, conf, resp_time, top5, df, generate_metrics_html()

        def load_history():
            try:
                df = pd.read_csv(classifier.history_file).tail(10)
                df = df.rename(columns={
                    "timestamp": "Horodatage", "image_name": "Image",
                    "predicted_class": "Classe prédite", "confidence": "Confiance (%)",
                    "top_5_predictions": "Top 5", "response_time_ms": "Temps (ms)",
                    "feedback": "Feedback"
                })
            except Exception:
                df = pd.DataFrame()
            return df, generate_metrics_html()

        def clear_and_refresh():
            msg = classifier.clear_history()
            try:
                df = pd.read_csv(classifier.history_file)
            except Exception:
                df = pd.DataFrame()
            return msg, df, generate_metrics_html()

        classify_btn.click(
            fn=classify_and_update,
            inputs=[input_image],
            outputs=[predicted_class, confidence, response_time, top_5_predictions, history_df, metrics_html]
        )

        btn_correct.click(
            fn=lambda: classifier.record_feedback("Correcte"),
            outputs=[feedback_status]
        )
        btn_incorrect.click(
            fn=lambda: classifier.record_feedback("Incorrecte"),
            outputs=[feedback_status]
        )

        refresh_btn.click(fn=generate_metrics_html, outputs=[metrics_html])

        clear_btn.click(
            fn=clear_and_refresh,
            outputs=[clear_status, history_df, metrics_html]
        )

        interface.load(fn=load_history, outputs=[history_df, metrics_html])

    return interface


if __name__ == "__main__":
    logger.info("Démarrage du système de classification d'images...")
    interface = create_interface()
    interface.launch(
        server_name=INTERFACE_CONFIG["server_name"],
        server_port=INTERFACE_CONFIG["server_port"],
        share=INTERFACE_CONFIG["share"],
        show_error=INTERFACE_CONFIG["show_error"]
    )
