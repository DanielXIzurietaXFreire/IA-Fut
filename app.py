# app.py
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ------------------ CARGAR MODELOS Y DATOS ------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "results.csv"

# Cargar artefactos
rf = joblib.load(MODEL_DIR / "random_forest_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
le = joblib.load(MODEL_DIR / "label_encoder_resultado.joblib")
elo = joblib.load(MODEL_DIR / "elo_ratings.joblib")
clasificados_2026 = joblib.load(MODEL_DIR / "clasificados_2026.joblib")
alias_raw = joblib.load(MODEL_DIR / "alias.joblib")
alias = {k.strip().lower(): v for k, v in alias_raw.items()}

# Cargar dataset para calcular forma reciente (opcional pero recomendado)
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# ------------------ FUNCIONES AUXILIARES ------------------
def buscar_equipo(nombre: str) -> str:
    """Normaliza el nombre del equipo usando alias."""
    nombre_clean = nombre.strip().lower()
    if nombre_clean in alias:
        return alias[nombre_clean]
    # Búsqueda aproximada
    for key, value in alias.items():
        if key in nombre_clean or nombre_clean in key:
            return value
    return nombre.title()

def calcular_forma(equipo: str):
    """Devuelve (avg_gf, avg_ga, form_ratio) últimos 5 partidos."""
    hoy = pd.Timestamp.now()
    df_team = df[(df['home_team'] == equipo) | (df['away_team'] == equipo)]
    df_team = df_team[df_team['date'] < hoy].sort_values('date').tail(5)
    if len(df_team) == 0:
        return 1.0, 1.0, 0.5  # valores por defecto

    gf, ga, pts = 0, 0, 0
    for _, row in df_team.iterrows():
        if row['home_team'] == equipo:
            gf += row['home_score']
            ga += row['away_score']
            if row['home_score'] > row['away_score']:
                pts += 3
            elif row['home_score'] == row['away_score']:
                pts += 1
        else:
            gf += row['away_score']
            ga += row['home_score']
            if row['away_score'] > row['home_score']:
                pts += 3
            elif row['away_score'] == row['home_score']:
                pts += 1
    n = len(df_team)
    return gf / n, ga / n, pts / (3 * n)

# ------------------ MODELO DE DATOS PARA REQUEST ------------------
class Sede(str, Enum):
    USA = "USA"
    Mexico = "Mexico"
    Canada = "Canada"

class PartidoRequest(BaseModel):
    equipo_local: str
    equipo_visitante: str
    sede: Sede

class PartidoResponse(BaseModel):
    equipo_local: str
    equipo_visitante: str
    sede: str
    elo_local: float
    elo_visitante: float
    forma_local: float
    forma_visitante: float
    probabilidades: dict   # {"H": 0.71, "D": 0.20, "A": 0.09}
    resultado_predicho: str  # "H", "D" o "A"
    mensaje: str = ""

# ------------------ ENDPOINTS ------------------
app = FastAPI(title="Predictor Mundial 2026", 
              description="API para predecir resultados de partidos del Mundial 2026")

@app.get("/")
def root():
    return {"message": "API Predictor Mundial 2026 - Usa POST /predict"}

@app.post("/predict", response_model=PartidoResponse)
def predict(partido: PartidoRequest):
    local = buscar_equipo(partido.equipo_local)
    visitante = buscar_equipo(partido.equipo_visitante)
    sede = partido.sede.value

    # Validar que ambos estén clasificados
    if local not in clasificados_2026 or visitante not in clasificados_2026:
        raise HTTPException(status_code=400, detail="Ambos equipos deben estar clasificados al Mundial 2026")

    # Obtener Elo
    elo_local = elo.get(local, 1500)
    elo_visitante = elo.get(visitante, 1500)

    # Ventaja local por sede
    if (sede == 'usa' and local == 'United States') or \
       (sede == 'mexico' and local == 'Mexico') or \
       (sede == 'canada' and local == 'Canada'):
        elo_local += 50
        is_neutral = 0
    else:
        is_neutral = 1

    # Calcular forma reciente
    avg_gf_local, avg_ga_local, form_local = calcular_forma(local)
    avg_gf_vis, avg_ga_vis, form_vis = calcular_forma(visitante)

    # Construir vector de características (mismo orden que en entrenamiento)
    X_new = np.array([[
        1.0,                     # importance (Mundial)
        is_neutral,
        elo_local, elo_visitante, elo_local - elo_visitante,
        avg_gf_local, avg_ga_local, form_local,
        avg_gf_vis, avg_ga_vis, form_vis
    ]])

    X_scaled = scaler.transform(X_new)
    pred = rf.predict(X_scaled)[0]
    probas = rf.predict_proba(X_scaled)[0]
    resultado = le.inverse_transform([pred])[0]

    # Construir respuesta
    class_index = {label: idx for idx, label in enumerate(le.classes_)}
    probabilidades = {
        "H": round(probas[class_index['H']] * 100, 1),
        "D": round(probas[class_index['D']] * 100, 1),
        "A": round(probas[class_index['A']] * 100, 1)
    }

    return PartidoResponse(
        equipo_local=local,
        equipo_visitante=visitante,
        sede=sede,
        elo_local=round(elo_local, 1),
        elo_visitante=round(elo_visitante, 1),
        forma_local=round(form_local, 3),
        forma_visitante=round(form_vis, 3),
        probabilidades=probabilidades,
        resultado_predicho=resultado,
        mensaje="Predicción generada correctamente"
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
