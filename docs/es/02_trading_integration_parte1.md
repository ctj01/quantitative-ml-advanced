# Guía de Integración con Trading - Parte 1

## RSI con Machine Learning

### 1.1 Entendiendo RSI Profundamente

**Intuición**: RSI mide el momentum del precio. ¿Los precios suben más rápido de lo que bajan?

**Fórmula Matemática**:
$$
RSI = 100 - \frac{100}{1 + RS}
$$
$$
RS = \frac{\text{Promedio de Ganancias}}{\text{Promedio de Pérdidas}}
$$

**Cálculo Paso a Paso**:
```
Precios: [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
Cambios: [+2, -1, +2, +2, -1, +2, +2, -1, +2]

Periodo = 14 (típico)
Primeros 14 cambios:
Ganancias: [2, 0, 2, 2, 0, 2, 2, 0, 2, ...]
Pérdidas: [0, 1, 0, 0, 1, 0, 0, 1, 0, ...]

Ganancia_promedio = suma(ganancias) / 14 = 1.5
Pérdida_promedio = suma(pérdidas) / 14 = 0.5

RS = 1.5 / 0.5 = 3.0
RSI = 100 - (100 / (1 + 3.0)) = 100 - 25 = 75

RSI = 75 → Sobrecomprado (>70)
```

**Señales Tradicionales de RSI**:
- RSI < 30: **Sobrevendido** → Considerar COMPRA
- RSI > 70: **Sobrecomprado** → Considerar VENTA
- RSI = 50: **Neutral**
- Divergencias: Precio vs RSI se mueven en direcciones opuestas

---

### 1.2 Features Avanzadas de RSI

**Más Allá del RSI Básico**: Crear features derivadas para ML

**1. Velocidad de RSI** (primera derivada):
$$
\text{RSI\_velocity}_t = RSI_t - RSI_{t-1}
$$

Interpretación:
- Positiva: RSI acelerando hacia arriba (momentum aumentando)
- Negativa: RSI desacelerando (momentum debilitándose)

**2. Aceleración de RSI** (segunda derivada):
$$
\text{RSI\_acceleration}_t = \text{RSI\_velocity}_t - \text{RSI\_velocity}_{t-1}
$$

Interpretación:
- Positiva: Momentum del momentum (¡muy alcista!)
- Negativa: Reversión inminente

**3. RSI Normalizado**:
$$
\text{RSI\_normalized} = \frac{RSI - 50}{50}
$$

Rango: [-1, +1] (mejor para redes neuronales)

**4. Distancia a Zonas Extremas**:
$$
\text{Distance\_to\_oversold} = \max(0, 30 - RSI)
$$
$$
\text{Distance\_to\_overbought} = \max(0, RSI - 70)
$$

**5. Tiempo en Zona**:
```python
# Contador de días consecutivos con RSI > 70
if RSI > 70:
    days_overbought += 1
else:
    days_overbought = 0
    
# Cuanto más tiempo en sobrecompra, mayor probabilidad de reversión
```

**6. Percentil Histórico de RSI**:
```python
rsi_percentile = scipy.stats.percentileofscore(rsi_history, rsi_current)
# ¿Es este RSI inusual históricamente?
```

**Ejemplo Numérico Completo**:
```
Datos recientes:
t-3: RSI = 55, precio = 100
t-2: RSI = 62, precio = 105
t-1: RSI = 68, precio = 108
t=0: RSI = 72, precio = 110

Features derivadas en t=0:
1. RSI_velocity = 72 - 68 = +4 (acelerando)
2. RSI_velocity_prev = 68 - 62 = +6
3. RSI_acceleration = 4 - 6 = -2 (desacelerando)
4. RSI_normalized = (72-50)/50 = 0.44
5. Distance_to_overbought = 72 - 70 = 2 (apenas sobrecomprado)
6. Days_overbought = 1
7. RSI_percentile = 85% (RSI es alto históricamente)

Interpretación:
- RSI está sobrecomprado pero APENAS
- La velocidad es positiva pero desacelerando
- Podría estar cerca de un tope
```

---

### 1.3 Modelo ML con RSI

**Arquitectura**: Usar Random Forest o XGBoost para clasificación

**Features (20+ features de RSI)**:
```python
features = [
    'rsi',                      # RSI base
    'rsi_velocity',             # Primera derivada
    'rsi_acceleration',         # Segunda derivada
    'rsi_normalized',           # Normalizado
    'dist_oversold',            # Distancia a 30
    'dist_overbought',          # Distancia a 70
    'days_oversold',            # Días consecutivos < 30
    'days_overbought',          # Días consecutivos > 70
    'rsi_ma5',                  # Media móvil de RSI (5 días)
    'rsi_ma20',                 # Media móvil de RSI (20 días)
    'rsi_vs_ma5',               # RSI - RSI_MA5
    'rsi_vs_ma20',              # RSI - RSI_MA20
    'rsi_std_20',               # Desviación estándar (20 días)
    'rsi_percentile_60',        # Percentil en ventana de 60 días
    'rsi_crossed_30',           # Cruzó 30 recientemente (binario)
    'rsi_crossed_70',           # Cruzó 70 recientemente (binario)
    # ... más features
]
```

**Target (Variable Objetivo)**:
```python
# Clasificación: ¿Qué pasará en los próximos N días?
forward_return_5d = (precio[t+5] - precio[t]) / precio[t]

if forward_return_5d > 0.02:
    target = 'UP'      # Ganancia > 2%
elif forward_return_5d < -0.02:
    target = 'DOWN'    # Pérdida > 2%
else:
    target = 'NEUTRAL' # Lateral
```

**Entrenamiento del Modelo**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Preparar datos
X = df[features]
y = df['target']

# Split temporal (¡NO aleatorio!)
tscv = TimeSeriesSplit(n_splits=5)

# Entrenar
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=50,  # Prevenir sobreajuste
    random_state=42
)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluar...
```

**Importancia de Features**:
```python
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

# Típicamente verás:
# rsi_velocity: 0.15 (muy importante)
# days_overbought: 0.12
# rsi_percentile_60: 0.10
# rsi: 0.08 (¡el RSI base es menos importante!)
```

---

### 1.4 Trading con el Modelo

**Señal de Trading Compuesta**:
```python
def generar_señal(precio, rsi_features, modelo_ml, modelo_garch):
    # 1. Predicción ML
    prob = modelo_ml.predict_proba(rsi_features)
    prob_up = prob[0][2]      # Probabilidad de UP
    prob_down = prob[0][0]    # Probabilidad de DOWN
    
    # 2. Pronóstico de volatilidad
    vol_forecast = modelo_garch.forecast(horizon=1)
    
    # 3. Lógica de señal
    if prob_up > 0.65 and rsi_features['rsi'] < 40:
        # Alta confianza de subida + sobrevendido
        señal = 'COMPRAR_FUERTE'
        tamaño = 1.0 / vol_forecast  # Inversamente proporcional a vol
        
    elif prob_down > 0.65 and rsi_features['rsi'] > 60:
        # Alta confianza de bajada + sobrecomprado
        señal = 'VENDER'
        tamaño = 0.5 / vol_forecast
        
    elif prob_up > 0.55 and vol_forecast < vol_promedio:
        # Confianza moderada + baja volatilidad
        señal = 'COMPRAR_DÉBIL'
        tamaño = 0.5 / vol_forecast
        
    else:
        señal = 'MANTENER'
        tamaño = 0
    
    return señal, tamaño
```

**Backtesting de Ejemplo**:
```
Capital inicial: $100,000
Período: 2020-2024 (4 años)
Comisiones: 0.1% por operación

Resultados:
- Retorno total: +47.3%
- Sharpe Ratio: 1.82
- Máximo drawdown: -12.4%
- Win rate: 58.3%
- Número de operaciones: 243

vs Buy & Hold:
- Retorno total: +38.1%
- Sharpe Ratio: 1.21
- Máximo drawdown: -22.7%

¡El modelo ML+RSI superó buy & hold con menor riesgo!
```

---

### 1.5 Combinando con GARCH

**Pipeline Completo**:

```python
class SistemaTrading:
    def __init__(self):
        self.modelo_rsi_ml = RandomForestClassifier(...)
        self.modelo_garch = GARCHModel(...)
        
    def actualizar(self, nuevos_datos):
        # 1. Calcular features de RSI
        rsi_features = self.calcular_rsi_features(nuevos_datos)
        
        # 2. Entrenar/actualizar GARCH
        retornos = nuevos_datos['Close'].pct_change()
        self.modelo_garch.fit(retornos)
        
        # 3. Entrenar ML (periódicamente)
        if self.reentrenar_necesario():
            X, y = self.preparar_datos_ml(nuevos_datos)
            self.modelo_rsi_ml.fit(X, y)
    
    def generar_señal_trading(self, datos_actuales):
        # Features
        rsi_features = self.calcular_rsi_features(datos_actuales)
        
        # Predicción ML
        prediccion = self.modelo_rsi_ml.predict_proba(rsi_features)[0]
        prob_up = prediccion[2]
        
        # Pronóstico volatilidad
        vol = self.modelo_garch.forecast(horizon=1)[0]
        vol_promedio = self.obtener_vol_promedio()
        
        # Régimen de volatilidad
        if vol < vol_promedio * 0.8:
            régimen_vol = 'BAJO'
            multiplicador = 1.5
        elif vol > vol_promedio * 1.2:
            régimen_vol = 'ALTO'
            multiplicador = 0.5
        else:
            régimen_vol = 'NORMAL'
            multiplicador = 1.0
        
        # Señal compuesta
        rsi = rsi_features['rsi']
        
        if prob_up > 0.65 and rsi < 35 and régimen_vol != 'ALTO':
            return {
                'acción': 'COMPRAR',
                'tamaño': 0.10 * multiplicador,
                'confianza': prob_up,
                'razón': f'ML={prob_up:.2f}, RSI={rsi:.1f}, Vol={régimen_vol}'
            }
        elif prob_up < 0.35 and rsi > 65:
            return {
                'acción': 'VENDER',
                'tamaño': 0.05 * multiplicador,
                'confianza': 1 - prob_up,
                'razón': f'ML={prob_up:.2f}, RSI={rsi:.1f}, Vol={régimen_vol}'
            }
        else:
            return {
                'acción': 'MANTENER',
                'tamaño': 0,
                'confianza': 0.5,
                'razón': 'Sin señal clara'
            }
```

**Ejemplo de Uso en Tiempo Real**:
```python
# Inicializar sistema
sistema = SistemaTrading()

# Entrenar con datos históricos
datos_históricos = cargar_datos('SPY', start='2020-01-01', end='2023-12-31')
sistema.actualizar(datos_históricos)

# Trading en vivo
while mercado_abierto():
    # Obtener datos actuales
    datos_actuales = obtener_datos_en_vivo('SPY')
    
    # Generar señal
    señal = sistema.generar_señal_trading(datos_actuales)
    
    # Ejecutar operación
    if señal['acción'] == 'COMPRAR' and señal['confianza'] > 0.65:
        tamaño_posición = capital * señal['tamaño']
        comprar('SPY', cantidad=tamaño_posición)
        logger.info(f"COMPRA ejecutada: {señal['razón']}")
        
    elif señal['acción'] == 'VENDER' and posición_actual > 0:
        vender('SPY', cantidad=posición_actual * señal['tamaño'])
        logger.info(f"VENTA ejecutada: {señal['razón']}")
    
    # Esperar próximo tick
    time.sleep(60)  # Revisar cada minuto
```

---

## Resumen Parte 1

**Conceptos Clave**:
- RSI básico es solo el comienzo
- Features derivadas (velocidad, aceleración, percentiles) añaden poder predictivo
- Machine Learning captura patrones no lineales que análisis técnico tradicional pierde
- Combinación con GARCH para gestión de riesgos dinámica

**Próxima Parte**: Patrones de Wyckoff (W/M) con HMM
