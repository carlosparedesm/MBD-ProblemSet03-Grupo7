
# a) Preparación de datos

## Importacion de librerias
import pandas as pd
import numpy as np

## 1)Carga de Bases de Datos
df_calendar = pd.read_csv("calendar.csv")
df_sales    = pd.read_csv("sales_train_evaluation.csv")
df_prices   = pd.read_csv("sell_prices.csv")

# =========================
# 2) Filtro Datos
#    - Departamento: FOODS_3
#    - Tiendas CA_1..CA_4
#    - Periodo entrenamiento: d_1..d_1900
# =========================
stores_ca = ["CA_1", "CA_2", "CA_3", "CA_4"]

df_sales_sub = df_sales[
    (df_sales["dept_id"] == "FOODS_3") &
    (df_sales["store_id"].isin(stores_ca))
].copy()

# =========================
# 3) Definición columnas train y test
# =========================
d_cols_train = [f"d_{i}" for i in range(1, 1901)]      # d_1..d_1900
d_cols_test  = [f"d_{i}" for i in range(1901, 1914)]   # d_1901..d_1913 (13 días)

id_cols = ["item_id", "store_id"]  # mínimo necesario

def melt_sales(df_wide, d_cols):
    out = df_wide[id_cols + d_cols].melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="y"
    )
    out["unique_id"] = out["store_id"] + "_" + out["item_id"]
    return out

df_long_train = melt_sales(df_sales_sub, d_cols_train)
df_long_test  = melt_sales(df_sales_sub, d_cols_test)

# =========================
# 4) Features desde calendar.csv (exógenas + ds)
# =========================
cal_cols = ["d", "date", "wm_yr_wk", "wday", "event_name_1", "event_name_2", "snap_CA"]
cal = df_calendar[cal_cols].copy()

cal["ds"] = pd.to_datetime(cal["date"])

# is_event = 1 si hay evento en cualquiera de las 2 columnas
cal["is_event"] = (
    cal["event_name_1"].notna() |
    cal["event_name_2"].notna()
).astype(int)

cal["day_of_week"] = cal["wday"]
cal["is_weekend"] = (cal["ds"].dt.dayofweek >= 5).astype(int)  # sábado/domingo

cal_keep = cal[["d", "ds", "wm_yr_wk", "day_of_week", "is_weekend", "snap_CA", "is_event"]]

df_long_train = df_long_train.merge(cal_keep, on="d", how="left")
df_long_test  = df_long_test.merge(cal_keep, on="d", how="left")

# =========================
# 5) Join con precios (sell_prices.csv)
#    Claves: store_id, item_id, wm_yr_wk
# =========================
prices_cols = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
prices = df_prices[prices_cols].copy()

df_long_train = df_long_train.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
df_long_test  = df_long_test.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

# forward-fill del precio por serie (recomendado)
df_long_train = df_long_train.sort_values(["unique_id", "ds"])
df_long_test  = df_long_test.sort_values(["unique_id", "ds"])

df_long_train["sell_price"] = df_long_train.groupby("unique_id")["sell_price"].ffill()
df_long_test["sell_price"]  = df_long_test.groupby("unique_id")["sell_price"].ffill()

# =========================
# 6) DataFrames finales MLForecast (train y test)
# =========================
df_train_mlf = df_long_train[[
    "unique_id", "ds", "y",
    "is_event", "snap_CA", "sell_price",
    "day_of_week", "is_weekend"
]].copy()

df_test_mlf = df_long_test[[
    "unique_id", "ds", "y",
    "is_event", "snap_CA", "sell_price",
    "day_of_week", "is_weekend"
]].copy()

# =========================
# 7) Checks + Entregable (head)
# =========================
print("TRAIN range:", df_train_mlf["ds"].min(), "->", df_train_mlf["ds"].max(), "| filas:", len(df_train_mlf))
print("TEST  range:", df_test_mlf["ds"].min(),  "->", df_test_mlf["ds"].max(),  "| filas:", len(df_test_mlf))

print("\nTRAIN head():")
print(df_train_mlf.head())

print("\nTEST head():")
print(df_test_mlf.head())

print("\nNulos (TRAIN) top:")
print(df_train_mlf.isna().mean().sort_values(ascending=False).head(10))

print("\nNulos (TEST) top:")
print(df_test_mlf.isna().mean().sort_values(ascending=False).head(10))


# b) Configuración del pipeline MLForecast

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

models = {
    "lgbm": LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    ),
    "xgb": XGBRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=8, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
}

fcst = MLForecast(
    models=models,
    freq="D",
    lags=[1, 7, 14, 28],
    lag_transforms={
        1: [RollingMean(window_size=7),
            RollingMean(window_size=14),
            RollingStd(window_size=7)]
    },
    date_features=[
        "day",
        "dayofyear",
        "week",
    ]
)

fcst.fit(
    df_train_mlf,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
    dropna=False,
    static_features=[] # Explicitly state that there are no static features
)

print("✅ Entrenamiento exitoso. Modelos:", list(fcst.models.keys()))


# c) Predicción y evaluación

# =========================
# Predicción y evaluación (13 días)
# Requiere:
# - fcst ya entrenado (punto 4.b)
# - df_train_mlf y df_test_mlf ya creados (punto 4.a)
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# 1) Predicción (h=13)
# -------------------------
preds = fcst.predict(h=13, X_df=df_test_mlf)

# preds suele venir como: unique_id | ds | lgbm | xgb
print("Preds head:")
print(preds.head())

# -------------------------
# 2) Dataset de evaluación (real vs pred)
# -------------------------
eval_df = (
    df_test_mlf[["unique_id", "ds", "y"]]
    .merge(preds, on=["unique_id", "ds"], how="left")
)

# Chequeo básico
if eval_df[["lgbm", "xgb"]].isna().any().any():
    na_cols = eval_df[["lgbm", "xgb"]].isna().mean().sort_values(ascending=False)
    print("\n⚠️ Hay nulos en predicciones. Proporción:")
    print(na_cols)

print("\nEval head:")
print(eval_df.head())

# -------------------------
# 3) Métricas (MAE, RMSE, MAPE seguro)
# -------------------------
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE excluyendo y_true == 0 para evitar división por cero.
    Retorna MAPE en porcentaje.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

metrics = []
for model in ["lgbm", "xgb"]:
    y_true = eval_df["y"].values
    y_hat  = eval_df[model].values

    mae  = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat)) # Corrected: Removed squared=False and added np.sqrt
    mape = safe_mape(y_true, y_hat)

    metrics.append({
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

metrics_df = pd.DataFrame(metrics).sort_values("MAE")
print("\nTabla comparativa de métricas:")
print(metrics_df)

# -------------------------
# 4) Gráfico
#    - últimos 60 días train
#    - real test
#    - predicciones ambos modelos
# -------------------------

# Elección serie con mayor venta total en entrenamiento (más representativa y menos ruido)
top_series = (
    df_train_mlf
    .groupby("unique_id")["y"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

print("\nSerie seleccionada para gráfico:", top_series)

train_plot = (
    df_train_mlf[df_train_mlf["unique_id"] == top_series]
    .sort_values("ds")
    .tail(60)
)

test_plot = (
    eval_df[eval_df["unique_id"] == top_series]
    .sort_values("ds")
)

plt.figure(figsize=(12, 5))

# Histórico train (últimos 60 días)
plt.plot(train_plot["ds"], train_plot["y"], label="Train (últimos 60 días)")

# Real test (13 días)
plt.plot(test_plot["ds"], test_plot["y"], label="Real (test)")

# Predicciones
plt.plot(test_plot["ds"], test_plot["lgbm"], linestyle="--", label="Forecast LGBM")
plt.plot(test_plot["ds"], test_plot["xgb"], linestyle="--", label="Forecast XGB")

plt.title(f"Forecast 13 días — {top_series}")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# d) Interpretación y decisión del negocio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from mlforecast.lag_transforms import RollingMean, RollingStd

# --- 1. Función para recrear las features ---
def generate_all_features(df, lags_config, lag_transforms_config, date_features_list):
    df = df.copy().sort_values(by=['unique_id', 'ds'])

    # Generate Lag features
    for lag in lags_config:
        df[f'lag{lag}'] = df.groupby('unique_id')['y'].shift(lag)

    # Generate Lag Transforms (Rolling features)
    for lag_val, transforms in lag_transforms_config.items():
        for transform in transforms:
            if isinstance(transform, RollingMean):
                df[f'rolling_mean_lag{lag_val}_window_{transform.window_size}'] = \
                    df.groupby('unique_id')[f'lag{lag_val}'].transform(lambda x: x.rolling(window=transform.window_size, min_periods=1).mean())
            elif isinstance(transform, RollingStd):
                df[f'rolling_std_lag{lag_val}_window_{transform.window_size}'] = \
                    df.groupby('unique_id')[f'lag{lag_val}'].transform(lambda x: x.rolling(window=transform.window_size, min_periods=1).std())

    # Generate Date features
    for feature_name in date_features_list:
        if feature_name == 'day':
            df['day'] = df['ds'].dt.day
        elif feature_name == 'dayofyear':
            df['dayofyear'] = df['ds'].dt.dayofyear
        elif feature_name == 'week':
            df['week'] = df['ds'].dt.isocalendar().week.astype(int)

    # Exogenous features (manually defined based on df_train_mlf setup)
    exog_features = ["is_event", "snap_CA", "sell_price", "day_of_week", "is_weekend"]
    # Add exogenous features to df directly, assuming they are already in the input df
    for feature in exog_features:
        if feature not in df.columns:
            # This case should ideally not happen if df_train_mlf was prepared correctly
            pass # Or add a warning/error

    # Drop rows with NaNs introduced by lag features
    df = df.dropna()
    return df

# --- 2. Recrear el DataFrame de entrenamiento con todas las features ---
# Usamos los parámetros del objeto fcst original para asegurar consistencia
lags_fcst = [1, 7, 14, 28] # Hardcoded from fcst initialization
lag_transforms_fcst = {1: [RollingMean(window_size=7), RollingMean(window_size=14), RollingStd(window_size=7)]} # Hardcoded
date_features_fcst = ["day", "dayofyear", "week"] # Hardcoded

df_train_augmented = generate_all_features(df_train_mlf.copy(), lags_fcst, lag_transforms_fcst, date_features_fcst)

# --- 3. Preparar X e y para el reentrenamiento de LightGBM ---
target_col = "y"
feature_cols = [col for col in df_train_augmented.columns if col not in ['unique_id', 'ds', target_col]]
X_refit = df_train_augmented[feature_cols]
y_refit = df_train_augmented[target_col]

# --- 4. Reentrenar el modelo LightGBM ---
# Usamos los mismos parámetros del LGBMRegressor que se usó en MLForecast
lgbm_refit = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm_refit.fit(X_refit, y_refit)

# --- 5. Extraer y mostrar Feature Importances ---
importance_df = pd.DataFrame({
    "feature": X_refit.columns,
    "importance": lgbm_refit.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 15 features más importantes (LightGBM - Refit):")
print(importance_df.head(15))

# --- 6. Graficar Feature Importances ---
top_n = 15
plot_df = importance_df.head(top_n).iloc[::-1]  # invertir para gráfico horizontal

plt.figure(figsize=(9, 6))
plt.barh(
    plot_df["feature"],
    plot_df["importance"]
)

plt.title("Top 15 Feature Importance — LightGBM (Refit sobre features MLForecast)")
plt.xlabel("Importancia (Gain)")
plt.ylabel("Feature")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()