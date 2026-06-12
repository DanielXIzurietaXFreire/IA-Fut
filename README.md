# Match Master API

API de predicción de resultados para el Mundial 2026 usando FastAPI.

## Estructura

- `app.py` - aplicación principal de FastAPI
- `requirements.txt` - dependencias Python
- `data/results.csv` - datos históricos para calcular forma reciente
- `models/` - artefactos del modelo entrenado

## Preparación local

1. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar localmente:
   ```bash
   python app.py
   ```
4. Probar la API:
   - `GET /`
   - `POST /predict`

Ejemplo de body para `/predict`:
```json
{
  "equipo_local": "United States",
  "equipo_visitante": "Mexico",
  "sede": "USA"
}
```

## GitHub

- Añadido `.gitignore` para ignorar entornos locales (`venv/`, `.venv/`, `env/`, `ENV/`).
- No incluir la carpeta `.venv/` ni archivos de configuración locales.
- Considera agregar una licencia si quieres compartir el repositorio públicamente.

## Despliegue en Render

1. Subir el repositorio a GitHub.
2. Crear un nuevo servicio en Render tipo `Web Service`.
3. Seleccionar `Python` y la rama del repositorio.
4. Configurar el comando de inicio:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
5. Render instalará dependencias desde `requirements.txt` automáticamente.

## Recomendaciones

- Si quieres mayor estabilidad, fija versiones en `requirements.txt`.
- Asegúrate de que `data/results.csv` y `models/` se suban al repositorio.
- Si los modelos son muy grandes, considera usar almacenamiento externo o LFS.
