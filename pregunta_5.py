

# =====================================================
# 0. CONFIGURACIÓN INICIAL
# =====================================================

# Manejo de datos
import pandas as pd
import numpy as np

# Modelos econométricos
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración de visualización
pd.set_option('display.float_format', '{:,.4f}'.format)

# =====================================================
# 1. CARGA DEL DATASET
# =====================================================

# El archivo debe estar subido previamente a Google Colab
df = pd.read_csv('online_retail_II.csv')

# Revisión inicial
df.head()
df.info()

# a) Preparación de datos para análisis de precios (5 pts) A

# =====================================================
# a) PREPARACIÓN Y LIMPIEZA DE DATOS
# =====================================================

# Copia de trabajo
data = df.copy()

# -------------------------------
# 1. Filtrar solo Reino Unido
# -------------------------------
data = data[data['Country'] == 'United Kingdom']

# -------------------------------
# 2. Eliminar devoluciones
#   - Quantity < 0
#   - Invoice que comienza con 'C'
# -------------------------------
data = data[
    (data['Quantity'] > 0) &
    (~data['Invoice'].astype(str).str.startswith('C'))
]

# -------------------------------
# 3. Eliminar registros inválidos
# -------------------------------
data = data[
    (data['Price'] > 0) &
    (data['Customer ID'].notna())
]

# -------------------------------
# 4. Top 10 productos más vendidos
# -------------------------------
top_10_products = (
    data
    .groupby(['StockCode', 'Description'])['Quantity']
    .sum()
    .reset_index()
    .rename(columns={'Quantity': 'ventas_totales'})
    .sort_values('ventas_totales', ascending=False)
    .head(10)
)

top_10_products


# -------------------------------
# 5. Agregación producto-semana
# -------------------------------

# Convertir fecha
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Crear semana
data['week'] = data['InvoiceDate'].dt.to_period('W').apply(lambda x: x.start_time)

# Filtrar solo top 10 productos
data_top = data[data['StockCode'].isin(top_10_products['StockCode'])]

# Agregar
weekly_data = (
    data_top
    .groupby(['StockCode', 'Description', 'week'])
    .agg(
        cantidad_total=('Quantity', 'sum'),
        precio_promedio=('Price', 'mean'),
        n_transacciones=('Invoice', 'nunique')
    )
    .reset_index()
)

weekly_data.head()


# =====================================================
# b) ELASTICIDAD NAIVE (LOG-LOG)
# =====================================================

elasticity_results = []

for stock, df_prod in weekly_data.groupby('StockCode'):

    df_prod = df_prod.copy()

    # Validación mínima de observaciones
    if len(df_prod) < 6:
        continue

    # Variables log
    df_prod['ln_q'] = np.log(df_prod['cantidad_total'])
    df_prod['ln_p'] = np.log(df_prod['precio_promedio'])

    # Modelo OLS
    model = smf.ols('ln_q ~ ln_p', data=df_prod).fit()

    elasticity_results.append({
        'StockCode': stock,
        'Description': df_prod['Description'].iloc[0],
        'beta': model.params['ln_p'],
        'p_value': model.pvalues['ln_p'],
        'R2': model.rsquared
    })

elasticity_table = pd.DataFrame(elasticity_results)
elasticity_table

# ¿Cuántos productos tienen elasticidad estadísticamente significativa (p < 0.05)? ¿Cuántos son elásticos (|𝛽| > 1) vs. inelásticos?
# Significancia estadística
significant = elasticity_table[elasticity_table['p_value'] < 0.05]

elastic = significant[abs(significant['beta']) > 1]
inelastic = significant[abs(significant['beta']) <= 1]

print(f"Elasticidades significativas (p < 0.05): {len(significant)}")
print(f"Productos elásticos (|β| > 1): {len(elastic)}")
print(f"Productos inelásticos (|β| ≤ 1): {len(inelastic)}")

# =====================================================
# c) MODELO CON EFECTOS FIJOS DE PRODUCTO Y MES
# =====================================================

# Crear variable de mes desde datetime (sin to_timestamp)
weekly_data['month'] = weekly_data['week'].dt.strftime('%Y-%m')

# Variables logarítmicas
weekly_data['ln_q'] = np.log(weekly_data['cantidad_total'])
weekly_data['ln_p'] = np.log(weekly_data['precio_promedio'])

# Limpieza post-log (buena práctica econométrica)
weekly_data = weekly_data.replace([np.inf, -np.inf], np.nan)
weekly_data = weekly_data.dropna(subset=['ln_q', 'ln_p'])

# Modelo pooled con efectos fijos de producto y mes
model_fe = smf.ols(
    formula='ln_q ~ ln_p + C(StockCode) + C(month)',
    data=weekly_data
).fit()

# Resumen del modelo
model_fe.summary()

# Comparación de coeficientes

beta_pooled = model_fe.params['ln_p']
beta_avg_naive = elasticity_table['beta'].mean()

print(f"β pooled (con efectos temporales): {beta_pooled:.4f}")
print(f"Promedio β individuales naive: {beta_avg_naive:.4f}")

# =====================================================
# d) RECOMENDACIÓN DE PRICING
# =====================================================

margen_actual = 0.40
margen_optimo = -1 / beta_pooled

print(f"Margen actual: {margen_actual:.2%}")
print(f"Margen óptimo (Lerner): {margen_optimo:.2%}")

if margen_optimo > margen_actual:
    print("Conclusión: El modelo sugiere SUBIR precios.")
else:
    print("Conclusión: El modelo sugiere BAJAR precios.")