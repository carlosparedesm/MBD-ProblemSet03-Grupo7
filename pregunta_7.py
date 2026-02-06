# Pregunta 7: Predecir precios

# =====================================================
# PARTE 0) LIBRERÍAS + CONFIG
# =====================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error

from lightgbm import LGBMRegressor

GRUPO_NUM = 4
PATH_XLSX = "Precios_PS03.xlsx"


# =====================================================
# PARTE 1) CARGA DE DATOS (detección robusta de hojas)
# =====================================================
xls = pd.ExcelFile(PATH_XLSX)
print("Hojas disponibles:", xls.sheet_names)

train_sheet = next(
    (s for s in xls.sheet_names if "train" in s.lower()),
    None
)

target_sheet = next(
    (s for s in xls.sheet_names if "target" in s.lower()),
    None
)

if train_sheet is None or target_sheet is None:
    raise ValueError(f"No encontré hojas train/target en: {xls.sheet_names}")

df_train = pd.read_excel(PATH_XLSX, sheet_name=train_sheet)
df_target = pd.read_excel(PATH_XLSX, sheet_name=target_sheet)

print("Train shape:", df_train.shape)
print("Target shape:", df_target.shape)


# =====================================================
# PARTE 2) LIMPIEZA BASE + DEFINICIÓN TARGET
# =====================================================
df_train.columns = df_train.columns.str.strip()
df_target.columns = df_target.columns.str.strip()

# --- columna objetivo ---
if "precio_uf" not in df_train.columns:
    cand = [c for c in df_train.columns if c.lower() == "precio_uf"]
    if cand:
        df_train.rename(columns={cand[0]: "precio_uf"}, inplace=True)
    else:
        raise ValueError("No encuentro columna 'precio_uf' en train")

# --- id requerido para output ---
if "titulo_propiedad" not in df_target.columns:
    cand = [c for c in df_target.columns if c.lower() == "titulo_propiedad"]
    if cand:
        df_target.rename(columns={cand[0]: "titulo_propiedad"}, inplace=True)
    else:
        raise ValueError("No encuentro 'titulo_propiedad' en target")

df_train = df_train.drop_duplicates()

y = df_train["precio_uf"].astype(float)
X = df_train.drop(columns=["precio_uf"])

titulo_target = df_target["titulo_propiedad"].copy()

# Alinear columnas
for col in X.columns:
    if col not in df_target.columns:
        df_target[col] = np.nan

X_target = df_target[X.columns].copy()


# =====================================================
# PARTE 3) PREPROCESAMIENTO
# =====================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
#    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])


# =====================================================
# PARTE 4) MODELO
# =====================================================
model = LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("lgbm", model)
])


# =====================================================
# PARTE 5) VALIDACIÓN CRUZADA (MAPE robusto)
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mapes = []

X_np = X.reset_index(drop=True)
y_np = y.values

for fold, (tr, va) in enumerate(kf.split(X_np), 1):
    X_tr, X_va = X_np.iloc[tr], X_np.iloc[va]
    y_tr, y_va = y_np[tr], y_np[va]

    y_tr_log = np.log1p(y_tr)
    pipe.fit(X_tr, y_tr_log)

    y_va_pred = np.expm1(pipe.predict(X_va))
    y_va_pred = np.clip(y_va_pred, 1e-6, None)

    # evitar división por cero en MAPE
    mask = y_va > 0
    fold_mape = mean_absolute_percentage_error(
        y_va[mask], y_va_pred[mask]
    )

    mapes.append(fold_mape)
    print(f"Fold {fold} MAPE: {fold_mape:.4f}")

print(f"\nMAPE CV promedio: {np.mean(mapes):.4f} | std: {np.std(mapes):.4f}")


# =====================================================
# PARTE 6) ENTRENAR MODELO FINAL + PREDECIR TARGET
# =====================================================
pipe.fit(X, np.log1p(y))

pred_target = np.expm1(pipe.predict(X_target))
pred_target = np.clip(pred_target, 0, None)

df_out = pd.DataFrame({
    "titulo_propiedad": titulo_target,
    "precio_uf": pred_target
})


# =====================================================
# PARTE 7) EXPORTAR RESULTADO
# =====================================================
out_name = f"Grupo_{GRUPO_NUM}_PS03Q07.xlsx"
df_out.to_excel(out_name, index=False)

print("\n✅ Archivo generado:", out_name)
print(df_out.head())

"""Luego del procesamiento de datos con el modelo LightGBM se obtienen los siguientes resultados:

*   Fold 1 MAPE: 0.1762
*   Fold 2 MAPE: 0.1636
*   Fold 3 MAPE: 0.1722
*   Fold 4 MAPE: 0.1717
*   Fold 5 MAPE: 0.1770

El MAPE promedio es de 17,21% lo que implica un error porcentual medio cercano al 17% lo que implica que las predicciones del precio en UF se desvian en esa proporción.

Adicionalmente podemos comentar que se obtiene una desviacion de 0,48% señalando que el modelo es estable y consistente sin variaciones relevantes. lo que sugiere una adecuada capacidad de generalización.

En el contexto de un problema de predicción de precios inmobiliarios, este nivel de error puede considerarse aceptable para análisis exploratorios y estimaciones de mercado, aunque podría requerir mejoras adicionales si se busca un nivel de precisión más alto para procesos de tasación o decisiones contractuales.

En conclusión, el modelo presenta un desempeño razonable y consistente, constituyendo una base sólida para futuras optimizaciones mediante ajustes de hiperparámetros, incorporación de nuevas variables explicativas o técnicas de calibración post-predicción.
"""