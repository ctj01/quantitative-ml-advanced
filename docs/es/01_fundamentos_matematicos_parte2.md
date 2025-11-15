# Fundamentos Matemáticos para ML Cuantitativo - Parte 2

## Modelado de Volatilidad

### 2.1 Proceso ARCH

**Intuición**: La volatilidad de hoy depende de los retornos al cuadrado de ayer - los grandes movimientos se agrupan.

**Explicación Geométrica**:
- La volatilidad viene en olas
- Períodos de calma → Períodos de caos → Períodos de calma
- "Agrupamiento de volatilidad" - observación clave de Mandelbrot

**Definición Matemática**:

Modelo ARCH(q):
$$
r_t = \sigma_t \varepsilon_t
$$
$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2
$$

donde:
- $r_t$ es el retorno en el tiempo $t$
- $\varepsilon_t \sim N(0,1)$ es la innovación estandarizada
- $\sigma_t^2$ es la varianza condicional
- $\omega$ es la varianza base
- $\alpha_i$ son coeficientes ARCH

**Restricciones de Parámetros**:
- $\omega > 0$ (varianza positiva)
- $\alpha_i \geq 0$ para todo $i$ (no negatividad)
- $\sum_{i=1}^{q} \alpha_i < 1$ (estacionariedad)

**Ejemplo Numérico - ARCH(1)**:
```
Parámetros: ω = 0.01, α = 0.3

Inicial: r_0 = 2%, σ_0² = 0.0004 (vol del 2%)

t=1: σ_1² = 0.01 + 0.3*(0.02)² = 0.01 + 0.00012 = 0.01012
     σ_1 = 10.06%
     ε_1 = -0.5 (shock aleatorio)
     r_1 = 10.06% * (-0.5) = -5.03%
     
t=2: σ_2² = 0.01 + 0.3*(-0.0503)² = 0.01 + 0.000759 = 0.010759
     σ_2 = 10.37%
     ε_2 = 1.2
     r_2 = 10.37% * 1.2 = 12.44%
     
t=3: σ_3² = 0.01 + 0.3*(0.1244)² = 0.01 + 0.00464 = 0.01464
     σ_3 = 12.10%
```

**Observación Clave**: ¡El gran shock negativo en t=1 aumentó la volatilidad en t=2!

**Aplicación en Trading**:

1. **Pronóstico de Volatilidad**:
```python
# Estimar ARCH con datos históricos
arch_model = ARCHModel(returns, lags=5)
arch_model.fit()

# Pronosticar volatilidad de mañana
vol_forecast = arch_model.forecast(horizon=1)

# Ajustar tamaño de posición
if vol_forecast > vol_histórica:
    reducir_posición()
```

2. **Gestión de Riesgos**:
- Calcular VaR (Valor en Riesgo)
- Establecer stop-loss dinámicos: `stop = precio - 2*σ_forecast`
- Dimensionamiento de posición: `tamaño ∝ 1/σ_forecast`

3. **Valoración de Opciones**:
- Usar volatilidad ARCH pronosticada en lugar de volatilidad histórica
- Mejor pricing para opciones a corto plazo

**Errores Comunes**:
- ❌ Usar ARCH para pronósticos a largo plazo (revierte rápido a la media)
- ❌ Ignorar efectos de apalancamiento (usar EGARCH)
- ❌ No verificar residuos para autocorrelación restante
- ❌ Aplicar a series no estacionarias sin diferenciar primero

---

### 2.2 Proceso GARCH

**Intuición**: GARCH añade "momentum" a la volatilidad - recuerda tanto shocks pasados COMO niveles de volatilidad pasados.

**¿Por qué GARCH > ARCH?**
- ARCH necesita muchos lags (parámetros) para capturar persistencia
- GARCH(1,1) a menudo supera a ARCH(5+) con menos parámetros
- Más parsimonioso y estable

**Definición Matemática**:

Modelo GARCH(p,q):
$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$

**Más Común**: GARCH(1,1)
$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Interpretación:
- $\omega$: volatilidad base de largo plazo
- $\alpha$: reacción a noticias (shocks recientes)
- $\beta$: persistencia (memoria de volatilidad)

**Varianza Incondicional** (estacionaria):
$$
\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}
$$

**Persistencia**: $\alpha + \beta$
- Cerca de 1 (ej. 0.98): muy persistente, decaimiento lento
- Lejos de 1 (ej. 0.70): reversión rápida a la media
- = 1: proceso IGARCH (volatilidad integrada, no estacionaria)

**Ejemplo Numérico - GARCH(1,1)**:
```
Parámetros típicos de mercado de valores:
ω = 0.00001, α = 0.08, β = 0.90
(Nota: α + β = 0.98 → altamente persistente)

Varianza incondicional: σ̄² = 0.00001/(1-0.98) = 0.0005 → σ̄ = 2.24%

Inicial: σ_0² = 0.0004 (vol del 2%)

t=1: r_1 = 3% (shock grande positivo)
     σ_1² = 0.00001 + 0.08*(0.03)² + 0.90*0.0004
          = 0.00001 + 0.000072 + 0.00036
          = 0.000442
     σ_1 = 2.10%

t=2: r_2 = -1% (shock pequeño negativo)
     σ_2² = 0.00001 + 0.08*(-0.01)² + 0.90*0.000442
          = 0.00001 + 0.000008 + 0.000398
          = 0.000416
     σ_2 = 2.04%

t=3: r_3 = 0.5% (shock pequeño)
     σ_3² = 0.00001 + 0.08*(0.005)² + 0.90*0.000416
          = 0.00001 + 0.000002 + 0.000374
          = 0.000386
     σ_3 = 1.96%
```

**Observa**: La volatilidad decae lentamente hacia σ̄ = 2.24%

**Estimación de Parámetros**:

Se usa Máxima Verosimilitud (MLE):
$$
\mathcal{L}(\theta) = \sum_{t=1}^{T} \log f(r_t | r_{t-1}, ..., r_1; \theta)
$$

donde la log-verosimilitud para cada observación:
$$
\log f(r_t) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma_t^2) - \frac{r_t^2}{2\sigma_t^2}
$$

**Aplicación Paso a Paso al Trading**:

**Paso 1: Recopilar Datos**
```python
import yfinance as yf
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')
returns = data['Close'].pct_change().dropna()
```

**Paso 2: Estimar GARCH**
```python
from arch import arch_model
model = arch_model(returns, vol='Garch', p=1, q=1)
fitted = model.fit()
print(fitted.summary())
# Obtienes: ω, α, β
```

**Paso 3: Pronosticar Volatilidad**
```python
forecast = fitted.forecast(horizon=5)
vol_forecast = np.sqrt(forecast.variance.values[-1])
# Array de volatilidades para los próximos 5 días
```

**Paso 4: Trading con Pronóstico**
```python
# Estrategia: Dimensionamiento de posición basado en volatilidad
vol_actual = vol_forecast[0]  # mañana
vol_promedio = returns.std()

if señal_RSI == 'COMPRAR':
    if vol_actual < vol_promedio:
        tamaño = capital * 0.10  # Posición normal
    else:
        tamaño = capital * 0.05  # Posición reducida (alta vol)
```

**Combinando GARCH con RSI**:
```python
def generar_señal_trading(precio, returns):
    # 1. Calcular RSI
    rsi = calcular_rsi(precio, periodo=14)
    
    # 2. Pronosticar volatilidad
    modelo_garch = ajustar_garch(returns)
    vol_pronóstico = modelo_garch.forecast(1)
    
    # 3. Lógica de señal compuesta
    if rsi < 30 and vol_pronóstico < percentil_50:
        return 'COMPRAR_FUERTE'  # Sobrevendido + Baja vol
    elif rsi < 30 and vol_pronóstico > percentil_75:
        return 'COMPRAR_DÉBIL'   # Sobrevendido + Alta vol (riesgoso)
    elif rsi > 70 and vol_pronóstico > percentil_75:
        return 'VENDER'          # Sobrecomprado + Alta vol
    else:
        return 'MANTENER'
```

---

### 2.3 EGARCH (GARCH Exponencial)

**Intuición**: Las malas noticias (retornos negativos) aumentan la volatilidad MÁS que las buenas noticias de la misma magnitud.

**¿Por qué EGARCH?**

Problema con GARCH estándar:
- Es **simétrico**: un retorno de +5% y -5% tienen el mismo impacto en la volatilidad
- Realidad del mercado: una caída del -5% causa más pánico/volatilidad que una ganancia del +5%
- Esto se llama "efecto de apalancamiento" o "efecto de asimetría"

**Definición Matemática**:

EGARCH(p,q):
$$
\log(\sigma_t^2) = \omega + \sum_{i=1}^{q} \left[\alpha_i \left|\frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right| + \gamma_i \frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right] + \sum_{j=1}^{p} \beta_j \log(\sigma_{t-j}^2)
$$

**EGARCH(1,1)** (más común):
$$
\log(\sigma_t^2) = \omega + \alpha \left|z_{t-1}\right| + \gamma z_{t-1} + \beta \log(\sigma_{t-1}^2)
$$

donde $z_t = \varepsilon_t / \sigma_t$ es la innovación estandarizada.

**Ventajas de EGARCH**:
1. ✅ **No negatividad automática**: $\log(\sigma^2)$ puede ser cualquier número real
2. ✅ **Captura asimetría**: parámetro $\gamma$
3. ✅ **Efectos de apalancamiento**: shocks negativos ↑ volatilidad más

**Interpretación del Parámetro γ**:
- $\gamma < 0$: Shocks negativos aumentan más la volatilidad (típico en acciones)
- $\gamma = 0$: Simétrico (como GARCH)
- $\gamma > 0$: Shocks positivos aumentan más la volatilidad (raro, algunos commodities)

**Ejemplo Numérico Detallado**:
```
Parámetros: ω = -0.10, α = 0.15, γ = -0.05, β = 0.95

Inicial: log(σ_0²) = -6.0 → σ_0² = 0.00248 → σ_0 = 4.98%

Escenario A: Shock positivo grande
r_1 = +7.5%, ε_1 = r_1/σ_0 = 7.5%/4.98% = 1.506
z_1 = 1.506

log(σ_1²) = -0.10 + 0.15*|1.506| + (-0.05)*1.506 + 0.95*(-6.0)
          = -0.10 + 0.2259 - 0.0753 - 5.70
          = -5.6494
σ_1² = 0.00352 → σ_1 = 5.93%

Incremento: 5.93% - 4.98% = +0.95%

Escenario B: Shock negativo grande
r_1 = -7.5%, ε_1 = -7.5%/4.98% = -1.506
z_1 = -1.506

log(σ_1²) = -0.10 + 0.15*|-1.506| + (-0.05)*(-1.506) + 0.95*(-6.0)
          = -0.10 + 0.2259 + 0.0753 - 5.70
          = -5.4988
σ_1² = 0.00408 → σ_1 = 6.39%

Incremento: 6.39% - 4.98% = +1.41%

¡El shock NEGATIVO aumentó la volatilidad más! (6.39% vs 5.93%)
Diferencia causada por el término γ*z
```

**Desglose del Efecto**:
- Componente $\alpha |z|$: **simétrico** (magnitud del shock)
- Componente $\gamma z$: **asimétrico** (dirección del shock)
- Para shock positivo: $\gamma z = -0.05 * 1.506 = -0.075$ (reduce efecto)
- Para shock negativo: $\gamma z = -0.05 * (-1.506) = +0.075$ (aumenta efecto)

**Aplicación en Trading**:

**1. Detección de Crash/Rally**:
```python
if γ < 0 and retorno < -2*σ:
    # Gran shock negativo detectado
    # Esperar aumento significativo de volatilidad
    aumentar_hedges()
    reducir_exposición()
```

**2. Gestión Dinámica de Riesgos**:
```python
# Después de movimiento del mercado
if retorno < 0:
    ajuste_vol = (abs(retorno) + γ*retorno) * α
else:
    ajuste_vol = (abs(retorno) + γ*retorno) * α

nuevo_stop = precio - 2.5 * σ_ajustada
```

**3. Trading de Opciones con EGARCH**:
```python
# Estrategia: Después de gran caída del mercado
if retorno_mercado < -3% and γ < 0:
    # EGARCH predice volatilidad aumentará significativamente
    # Comprar opciones PUT o call straddles
    señal = 'COMPRAR_VOLATILIDAD'
    
elif retorno_mercado > +3%:
    # EGARCH predice volatilidad aumentará menos
    # Vender opciones (recoger prima)
    señal = 'VENDER_VOLATILIDAD'
```

**4. Combinación con Divergencias**:
```python
# Detectar divergencia bajista
divergencia_bajista = (precio_nuevo_alto and rsi_menor)

if divergencia_bajista and γ < 0:
    # Si precio cae, EGARCH predice gran aumento de vol
    # Cerrar posiciones largas, comprar puts
    estrategia = 'PROTECCIÓN_BAJISTA'
```

**Comparación GARCH vs EGARCH**:

```
Shock de +5%:
- GARCH: σ aumenta X%
- EGARCH: σ aumenta X - δ%

Shock de -5%:
- GARCH: σ aumenta X% (igual)
- EGARCH: σ aumenta X + δ% (más)

donde δ depende de γ
```

**Cuándo Usar Cada Modelo**:
- **ARCH**: Datos simples, pocas observaciones, exploración inicial
- **GARCH**: Mayoría de aplicaciones financieras, balance precisión/complejidad
- **EGARCH**: Mercados de acciones con fuerte efecto de apalancamiento

---

## Resumen de Parte 2

Has aprendido modelos de volatilidad avanzados:
- **ARCH**: Volatilidad depende de shocks pasados al cuadrado
- **GARCH**: Añade persistencia (memoria de volatilidad)
- **EGARCH**: Captura asimetría (malas noticias ≠ buenas noticias)

**Aplicaciones Clave**:
- Pronóstico de volatilidad para gestión de riesgos
- Dimensionamiento dinámico de posiciones
- Trading de opciones basado en volatilidad
- Combinación con indicadores técnicos (RSI, divergencias)

**Próxima Parte**: Modelos Ocultos de Markov y detección de régimen
