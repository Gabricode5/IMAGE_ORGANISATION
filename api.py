from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
from io import BytesIO
from main import ImageClassifier

app = FastAPI(title="Classification d'Images — API")
classifier = ImageClassifier()


@app.get("/health")
def health():
    """Vérifie que l'API fonctionne correctement."""
    return {"statut": "ok", "modele": "ResNet50"}


@app.post("/predict")
async def predict(fichier: UploadFile = File(...)):
    """
    Reçoit une image et retourne la prédiction du modèle.

    INCIDENT #001 — Bug corrigé :
    Avant correction, la ligne suivante était :
        img = img.resize((64, 64))
    Cela causait une erreur car ResNet50 attend des images de taille 224x224.
    Correction : la taille est maintenant lue depuis MODEL_CONFIG dans config.py.
    """
    contenu = await fichier.read()
    img = Image.open(BytesIO(contenu))

    predicted_class, confidence, top_5, response_time_ms = classifier.classify_image(img)

    return {
        "classe": predicted_class,
        "confiance_pct": round(confidence, 2),
        "top_5": top_5,
        "temps_ms": round(response_time_ms, 0)
    }


@app.get("/metrics")
def metrics():
    """Retourne les métriques de monitoring de l'application."""
    total, errors, low_conf, avg_conf, avg_time, correct_fb, incorrect_fb = classifier.get_metrics()
    return {
        "predictions_totales": total,
        "erreurs": errors,
        "confiance_faible": low_conf,
        "confiance_moyenne_pct": avg_conf,
        "temps_reponse_moyen_ms": avg_time,
        "feedback_correct": correct_fb,
        "feedback_incorrect": incorrect_fb
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
