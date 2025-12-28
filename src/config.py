import os

# --- PATHS ---
RAW_DATA_DIR = os.path.join("data", "raw")
FEATURE_DIR = os.path.join("data", "features")
MODEL_DIR = "models"

# --- DIMENSIONS (Approach 2) ---
# The components of the Unified Feature Vector
DIM_WHISPER = 1280
DIM_MS_CLAP = 2048
DIM_LAION = 1536
DIM_DEBERTA = 768

# The final vector size after concatenating embeddings + geometric injections
# (Base 5632 + Element-wise interactions + Scalar metrics)
TOTAL_DIM = 9220 

# --- HYPERPARAMETERS ---
# The exact ensemble weights that achieved SRCC 0.653
WEIGHT_XGBOOST = 0.56
WEIGHT_SVR = 0.44

# Stream A: XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'random_state': 42
}

# Stream B: SVR Parameters
SVR_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'epsilon': 0.1
}
