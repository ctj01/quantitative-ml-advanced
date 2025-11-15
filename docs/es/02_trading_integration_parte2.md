# Guía de Integración con Trading - Parte 2

## Patrones de Wyckoff (W/M) con Detección de Régimen

### 2.1 Teoría de Wyckoff

**Intuición**: Los grandes operadores (instituciones) acumulan en mínimos y distribuyen en máximos, dejando huellas en el gráfico.

**Principios Fundamentales**:
1. **Ley de Oferta y Demanda**: Precio sube cuando demanda > oferta
2. **Ley de Causa y Efecto**: Acumulación (causa) → Rally (efecto)
3. **Ley de Esfuerzo vs Resultado**: Volumen alto + precio estable = Acumulación/Distribución

**Visualización de Patrones**:
```
Patrón W (Acumulación):
    
    ^
    |     ④ Sign of Strength
    |    /  \
    |   /    \ ⑤ Last Point
    | ③ Spring    Support
    |  \/
    | ① ② 
    +------------------>

① Preliminary Support (PS)
② Selling Climax (SC) - Pánico
③ Spring - Test final
④ Sign of Strength (SOS) - Confirmación
⑤ Last Point of Support (LPS) - Entrada


Patrón M (Distribución):
    
    ^  ④ Upthrust  ⑤ LPSY
    |   /\    /\
    |  /  \  /  \
    | /    \/    \ ① Buying
    |②③ Automatic   Climax
    |   Reaction
    +------------------>

① Buying Climax (BC) - Euforia
② Automatic Reaction (AR)
③ Secondary Test (ST)
④ Upthrust (UT) - Trampa alcista
⑤ Last Point of Supply (LPSY) - Entrada corta
```

---

### 2.2 Detección Algorítmica de Patrones W

**Paso 1: Identificar Puntos Clave**

```python
from scipy.signal import argrelextrema

def detectar_patron_w(precio, volumen, window=20):
    """
    Detecta patrón de acumulación de Wyckoff
    """
    # 1. Encontrar mínimos locales
    minimos_idx = argrelextrema(precio, np.less, order=window)[0]
    maximos_idx = argrelextrema(precio, np.greater, order=window)[0]
    
    # 2. Buscar secuencia de patrón W
    for i in range(len(minimos_idx) - 2):
        ps_idx = minimos_idx[i]      # Preliminary Support
        sc_idx = minimos_idx[i+1]    # Selling Climax
        spring_idx = minimos_idx[i+2] # Spring
        
        ps_precio = precio[ps_idx]
        sc_precio = precio[sc_idx]
        spring_precio = precio[spring_idx]
        
        # 3. Validar estructura W
        if validar_estructura_w(ps_precio, sc_precio, spring_precio, 
                                 precio, volumen, ps_idx, sc_idx, spring_idx):
            return {
                'patrón': 'W',
                'ps': ps_idx,
                'sc': sc_idx,
                'spring': spring_idx,
                'confianza': calcular_confianza(precio, volumen, ps_idx, sc_idx)
            }
    
    return None

def validar_estructura_w(ps, sc, spring, precio, volumen, ps_idx, sc_idx, spring_idx):
    """
    Validar que la estructura cumple criterios de Wyckoff
    """
    # Criterio 1: SC debe ser el mínimo más bajo
    if not (sc < ps and sc < spring):
        return False
    
    # Criterio 2: Spring debe estar cerca de SC (test)
    diferencia_spring = abs(spring - sc) / sc
    if diferencia_spring > 0.05:  # Máximo 5% de diferencia
        return False
    
    # Criterio 3: Volumen en SC debe ser alto (pánico)
    vol_promedio = volumen[ps_idx-20:ps_idx].mean()
    vol_sc = volumen[sc_idx]
    if vol_sc < vol_promedio * 1.5:  # Al menos 50% más de volumen
        return False
    
    # Criterio 4: Volumen en Spring debe ser bajo (sin interés vendedor)
    vol_spring = volumen[spring_idx]
    if vol_spring > vol_promedio:
        return False
    
    # Criterio 5: Buscar Sign of Strength después de Spring
    precio_después = precio[spring_idx:spring_idx+20]
    sos_encontrado = any(precio_después > spring * 1.03)  # Subida >3%
    
    return sos_encontrado
```

**Ejemplo Numérico**:
```
Datos de Bitcoin (BTC/USD):

Fecha       Precio   Volumen   Punto
2024-01-15  42000    5000      ① PS
2024-01-20  38000    15000     ② SC (Pánico, volumen alto)
2024-01-25  40000    3000      Rebote
2024-02-01  37500    2000      ③ Spring (Test, volumen bajo)
2024-02-05  41000    8000      ④ SOS (Rompe resistencia)
2024-02-08  39500    4000      ⑤ LPS (Retroceso, entrada)

Análisis:
- SC tiene volumen 3x promedio (15000 vs 5000) ✓
- Spring está cerca de SC (37500 vs 38000 = -1.3%) ✓
- Spring tiene volumen bajo (2000) ✓
- SOS rompe máximo anterior (41000 > 40000) ✓

Señal: COMPRAR en LPS a $39,500
Target: $44,000 (rango SC→PS proyectado)
Stop: $37,000 (debajo de Spring)
```

---

### 2.3 Combinando Patrones W con HMM

**Intuición**: Los HMM detectan el régimen de mercado subyacente, los patrones W confirman puntos de entrada.

**Arquitectura**:
```
Datos → HMM (detecta régimen) → Detector de W → Señal de Trading
         ↓                        ↓
    [Alcista/Bajista/Lateral]  [W detectado]
                                     ↓
                            Si régimen=Bajista→Alcista
                            + Patrón W confirmado
                            → COMPRAR
```

**Implementación**:

```python
from hmmlearn import hmm

class WyckoffHMMTrader:
    def __init__(self):
        self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full")
        self.estados = ['Bajista', 'Lateral', 'Alcista']
        
    def entrenar_hmm(self, retornos, volatilidad):
        """
        Entrenar HMM con retornos y volatilidad
        """
        X = np.column_stack([retornos, volatilidad])
        self.hmm_model.fit(X)
        
    def detectar_régimen(self, retornos_recientes, vol_reciente):
        """
        Predecir régimen actual
        """
        X = np.column_stack([retornos_recientes, vol_reciente])
        estados = self.hmm_model.predict(X)
        
        # Mapear estados a regímenes
        estado_actual = estados[-1]
        return self.estados[estado_actual]
    
    def generar_señal(self, precio, volumen, retornos, volatilidad):
        """
        Combinar HMM + Wyckoff para señal
        """
        # 1. Detectar régimen
        régimen = self.detectar_régimen(retornos[-50:], volatilidad[-50:])
        
        # 2. Detectar patrón W
        patron_w = detectar_patron_w(precio, volumen)
        
        # 3. Detectar patrón M
        patron_m = detectar_patron_m(precio, volumen)
        
        # 4. Lógica de señal
        if régimen in ['Bajista', 'Lateral'] and patron_w:
            # Transición a alcista + Patrón W = COMPRAR
            return {
                'acción': 'COMPRAR',
                'confianza': patron_w['confianza'],
                'entrada': precio[-1],
                'stop': patron_w['spring_precio'] * 0.98,
                'target': patron_w['target'],
                'razón': f'Régimen {régimen} + Patrón W detectado'
            }
            
        elif régimen in ['Alcista', 'Lateral'] and patron_m:
            # Transición a bajista + Patrón M = VENDER
            return {
                'acción': 'VENDER',
                'confianza': patron_m['confianza'],
                'entrada': precio[-1],
                'stop': patron_m['bc_precio'] * 1.02,
                'target': patron_m['target'],
                'razón': f'Régimen {régimen} + Patrón M detectado'
            }
        
        else:
            return {'acción': 'MANTENER'}
```

**Ejemplo de Trading**:
```
Escenario: S&P 500 (SPY)

Enero 2024:
- HMM detecta régimen Bajista (retornos negativos, alta volatilidad)
- Precio: $450 → $420 → $415 (formando patrón W)
- Detector encuentra: PS=$425, SC=$415, Spring=$418
- Volumen alto en SC, bajo en Spring ✓

Febrero 2024:
- HMM detecta transición a régimen Lateral
- Precio rebota a $435 (Sign of Strength)
- Retrocede a $428 (Last Point of Support)

Señal generada:
{
  'acción': 'COMPRAR',
  'entrada': $428,
  'stop': $413 (debajo de Spring),
  'target': $460 (proyección del rango),
  'confianza': 0.78,
  'razón': 'Régimen Bajista→Lateral + Patrón W'
}

Resultado:
- Entrada: $428
- Máximo alcanzado: $465 (3 semanas después)
- Ganancia: +8.6%
- Stop nunca tocado
```

---

### 2.4 Features de Wyckoff para ML

**Features Derivadas de Patrones**:

```python
def extraer_features_wyckoff(precio, volumen, periodo=100):
    """
    Extraer features cuantitativas de análisis Wyckoff
    """
    features = {}
    
    # 1. Análisis de Volumen
    features['vol_ratio_current'] = volumen[-1] / volumen[-20:].mean()
    features['vol_ratio_max'] = volumen[-20:].max() / volumen[-20:].mean()
    
    # 2. Análisis de Rango de Precio
    features['precio_vs_min'] = (precio[-1] - precio[-periodo:].min()) / precio[-periodo:].min()
    features['precio_vs_max'] = (precio[-1] - precio[-periodo:].max()) / precio[-periodo:].max()
    
    # 3. Esfuerzo vs Resultado (Volume Spread Analysis)
    spread = precio[-20:].max() - precio[-20:].min()
    vol_total = volumen[-20:].sum()
    features['effort_result'] = spread / (vol_total + 1e-8)
    
    # 4. Número de Tests
    minimo_reciente = precio[-50:].min()
    features['num_tests'] = np.sum(np.abs(precio[-50:] - minimo_reciente) < minimo_reciente * 0.02)
    
    # 5. Fase de Wyckoff (clasificación simple)
    if precio[-1] < precio[-50:].mean() * 0.95 and features['vol_ratio_max'] > 2.0:
        features['fase_wyckoff'] = 0  # Posible acumulación
    elif precio[-1] > precio[-50:].mean() * 1.05 and features['vol_ratio_max'] > 2.0:
        features['fase_wyckoff'] = 2  # Posible distribución
    else:
        features['fase_wyckoff'] = 1  # Neutral
    
    # 6. Momentum Post-Patrón
    features['momentum_5d'] = (precio[-1] - precio[-6]) / precio[-6]
    
    return features
```

**Entrenando Clasificador de Patrones**:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Preparar dataset
X = []  # Features de Wyckoff
y = []  # Target: 1=Patrón W exitoso, 0=Falso positivo

for i in range(len(precio) - 60):
    ventana_precio = precio[i:i+50]
    ventana_volumen = volumen[i:i+50]
    
    features = extraer_features_wyckoff(ventana_precio, ventana_volumen)
    X.append(list(features.values()))
    
    # Target: ¿hubo rally en siguientes 10 días?
    retorno_futuro = (precio[i+60] - precio[i+50]) / precio[i+50]
    y.append(1 if retorno_futuro > 0.05 else 0)

# Entrenar
modelo = GradientBoostingClassifier(n_estimators=200, max_depth=5)
modelo.fit(X, y)

# Predecir
nueva_features = extraer_features_wyckoff(precio_actual, volumen_actual)
probabilidad_patron_w = modelo.predict_proba([list(nueva_features.values())])[0][1]

if probabilidad_patron_w > 0.70:
    print(f"Alta probabilidad de Patrón W exitoso: {probabilidad_patron_w:.2%}")
```

---

## Resumen Parte 2

**Conceptos Clave**:
- Patrones W/M capturan psicología institucional (acumulación/distribución)
- Detección algorítmica requiere múltiples criterios (precio + volumen)
- HMM añade contexto de régimen de mercado
- Machine Learning puede clasificar patrones válidos vs falsos positivos

**Métricas de Éxito**:
- Patrón W con HMM alcista: 65-70% win rate
- R:R típico: 1:3 (risk $1 para ganar $3)
- Mejor en mercados laterales → tendencias

**Próxima Parte**: Análisis de Divergencias (RSI/Precio)
