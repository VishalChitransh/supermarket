# SuperKart Sales Forecasting

A reproducible machine-learning project for estimating product-store revenue from supplied SuperKart sales history. It includes an exploratory notebook, a shared preprocessing and training package, a Flask inference API, a Streamlit interface, and a two-container Docker deployment configured for GitHub Codespaces.

## Project contents

- `data/SuperKart.csv`: training data supplied with the project.
- `data/Batch_Data_SuperKart.csv`: example batch input using the API feature schema.
- `notebooks/Full_Code_SuperKart_Model_Deployment_Notebook.ipynb`: completed project notebook.
- `src/superkart/`: feature engineering, model training, and API.
- `frontend/app.py`: online and batch prediction interface.
- `docker-compose.yml`: Flask + Streamlit services.

The target is `Product_Store_Sales_Total`. The model uses the ten engineered fields listed in `MODEL_FEATURE_COLUMNS`. Product ID and store ID are not passed through as raw identifiers. Product ID prefix, store age, and a perishables grouping are derived with the same rules for training and batch inference. Store age uses the supplied example's 2025 reference year so an establishment year of 2009 corresponds to age 16.

## Local setup

From this directory, create and activate a virtual environment, then install the package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[api,frontend,notebook,dev]"
```

Train and serialize the model (GridSearchCV can take several minutes):

```powershell
python -m superkart.train
```

For a quicker baseline run, skip hyperparameter tuning with `python -m superkart.train --no-tune`.

Run the API and UI in separate terminals:

```powershell
python -m superkart.api
```

```powershell
$env:API_URL = "http://localhost:7860"
streamlit run frontend/app.py
```

Open `http://localhost:8501`. The API provides `GET /healthz`, `POST /v1/predict` (JSON), and `POST /v1/predictbatch` (multipart CSV in the `file` field). Batch data must contain the ten engineered model columns; see `data/Batch_Data_SuperKart.csv`.

## Tests

```powershell
python -m pytest -q
```

## GitHub Codespaces deployment

1. Push this project to the GitHub repository you choose and create a Codespace using the included `.devcontainer/devcontainer.json`.
2. Both Docker images install their runtime dependencies from the root `requirements.txt`. The backend image trains and stores a baseline model from the included data during its build. The notebook additionally demonstrates cross-validated tuning and saves the selected model locally; generated artifacts remain git-ignored.
3. The compose stack starts Flask on port 7860 and Streamlit on port 8501. Set port 7860 to Public in the Codespaces Ports panel only when external API access is required.
4. Configure Streamlit's `API_URL` as `http://backend:7860` inside Docker. No personal access token is stored in this project.

No GitHub repository URL or deployment credential was supplied, so publishing and remote deployment are not performed by this local scaffold.
