# ============================================
# LIBRERÍAS Y CONFIGURACIÓN
# ============================================

import pandas as pd
import numpy as np

pd.set_option('display.float_format', '{:,.2f}'.format)

# ============================================
# DATASET DE RIDEFLOW
# ============================================

data = {
    "Segmento": ["Ejecutivos", "Casual", "Nocturno", "Aeropuerto"],
    "Descripcion": [
        "Viajes L-V 7-9am, zonas empresariales",
        "Viajes fines de semana, zonas comerciales",
        "Viajes Vie-Sáb 23:00-4:00",
        "Viajes hacia/desde SCL"
    ],
    "Viajes_dia": [12500, 17500, 10000, 10000],
    "Elasticidad": [-0.6, -1.8, -1.2, -0.4]
}

df = pd.DataFrame(data)

# Parámetros generales
precio_base = 3500
costo_variable = 2100

# Validaciones de datos, assert es una instrucción de control.
assert all(df["Viajes_dia"] > 0), "Error: viajes diarios deben ser positivos"
assert all(df["Elasticidad"] < 0), "Error: elasticidades deben ser negativas"
assert precio_base > costo_variable, "Error: el precio debe ser mayor al costo"

df

# a.1 Margen de contribución actual (%)
margen_actual = (precio_base - costo_variable) / precio_base

margen_actual

# a.2 Contribución diaria total con precio uniforme
contribucion_por_viaje = precio_base - costo_variable
viajes_totales = df["Viajes_dia"].sum()
contribucion_diaria_total = contribucion_por_viaje * viajes_totales

contribucion_diaria_total

# a.3 Margen óptimo por segmento (Regla de Lerner)
# (P - C) / P = 1 / |elasticidad|
df["Margen_optimo_Lerner"] = 1 / df["Elasticidad"].abs()

df

# Clasificación: subexplotado vs sobreexplotado
df["Estado_margen"] = np.where(
    margen_actual < df["Margen_optimo_Lerner"],
    "Subexplotado",
    "Sobreexplotado"
)

df

# ------------------------------------------------------------
# b) DISEÑO DE PRICING DINÁMICO (6 pts)
# ------------------------------------------------------------

# b.1 Precio óptimo por segmento
# P* = C * |β| / (|β| - 1)
df["Precio_optimo"] = costo_variable * (
    df["Elasticidad"].abs() / (df["Elasticidad"].abs() - 1)
)

df

# b.2 Multiplicador respecto al precio base
df["Multiplicador_precio"] = df["Precio_optimo"] / precio_base

df

# b.3 Nueva contribución diaria por segmento
# (Se asume demanda constante en el corto plazo)
df["Contribucion_por_viaje_opt"] = df["Precio_optimo"] - costo_variable
df["Contribucion_diaria_opt"] = (
    df["Contribucion_por_viaje_opt"] * df["Viajes_dia"]
)

df

# b.4 Contribución total con pricing dinámico
contribucion_total_optima = df["Contribucion_diaria_opt"].sum()

# Incremento porcentual vs precio uniforme
incremento_porcentual = (
    (contribucion_total_optima - contribucion_diaria_total)
    / contribucion_diaria_total
) * 100

incremento_porcentual