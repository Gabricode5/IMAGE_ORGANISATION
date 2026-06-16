from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
from PIL import Image
from io import BytesIO
from main import ImageClassifier
from config import FILE_CONFIG, LOGGING_CONFIG

app = FastAPI(title="Classification d'Images — API")
classifier = ImageClassifier()


@app.get("/health")
def health():
    """Vérifie l'état réel des composants critiques de l'application."""
    checks = {}
    erreurs = []

    # Vérification : modèle chargé en mémoire
    checks["model_loaded"] = classifier.model is not None
    if not checks["model_loaded"]:
        erreurs.append("Le modèle ResNet50 n'est pas chargé en mémoire")

    # Vérification : fichier CSV accessible en lecture/écriture
    try:
        with open(FILE_CONFIG["history_file"], "r+", encoding="utf-8"):
            pass
        checks["history_file_accessible"] = True
    except Exception:
        checks["history_file_accessible"] = False
        erreurs.append(f"Fichier CSV inaccessible : {FILE_CONFIG['history_file']}")

    # Vérification : fichier de log accessible en écriture
    try:
        with open(LOGGING_CONFIG["file"], "a", encoding="utf-8"):
            pass
        checks["log_file_writable"] = True
    except Exception:
        checks["log_file_writable"] = False
        erreurs.append(f"Fichier log inaccessible : {LOGGING_CONFIG['file']}")

    if erreurs:
        return JSONResponse(
            status_code=503,
            content={
                "statut": "dégradé",
                "modele": "ResNet50",
                "checks": checks,
                "erreurs": erreurs
            }
        )

    return {"statut": "ok", "modele": "ResNet50", "checks": checks}


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
    total, errors, low_conf, avg_conf, avg_time, correct_fb, incorrect_fb, consec_errors = classifier.get_metrics()
    taux_erreur = round((errors / total * 100), 2) if total > 0 else 0.0
    return {
        "predictions_totales": total,
        "erreurs": errors,
        "taux_erreur_pct": taux_erreur,
        "erreurs_consecutives_courantes": consec_errors,
        "confiance_faible": low_conf,
        "confiance_moyenne_pct": avg_conf,
        "temps_reponse_moyen_ms": avg_time,
        "feedback_correct": correct_fb,
        "feedback_incorrect": incorrect_fb
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
