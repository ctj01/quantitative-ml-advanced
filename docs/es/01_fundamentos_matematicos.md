# Fundamentos Matemáticos para ML Cuantitativo

## Tabla de Contenidos
1. [Matemáticas de Series Temporales](#matemáticas-de-series-temporales)
2. [Procesos Estocásticos](#procesos-estocásticos)
3. [Modelado de Volatilidad](#modelado-de-volatilidad)
4. [Modelos Ocultos de Markov](#modelos-ocultos-de-markov)
5. [Teoría de Optimización](#teoría-de-optimización)
6. [Teoría de la Información](#teoría-de-la-información)

---

## Matemáticas de Series Temporales

### 1.1 Procesos Autoregresivos

**Intuición**: Un proceso AR usa valores pasados para predecir valores futuros, como el precio de ayer influye en el precio de hoy.

**Explicación Geométrica**: 
- Imagina una banda elástica conectando puntos consecutivos en una serie temporal
- El proceso AR crea "memoria" - jalando valores futuros hacia patrones pasados
- Órdenes AR más altas capturan oscilaciones más complejas

**Definición Matemática**:

Un proceso AR(p) se define como:
$$
X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \varepsilon_t
$$

donde:
- $X_t$ es el valor en el tiempo $t$
- $c$ es una constante
- $\phi_i$ son coeficientes autoregresivos
- $\varepsilon_t \sim N(0, \sigma^2)$ es ruido blanco

**Condición de Estacionariedad**:
Las raíces del polinomio característico deben estar fuera del círculo unitario:
$$
1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p = 0 \implies |z| > 1
$$

**Ejemplo Numérico - AR(1)**:
```
Dado: X_t = 0.7*X_{t-1} + ε_t, X_0 = 10, σ = 1

t=0: X_0 = 10.00
t=1: X_1 = 0.7*10 + ε_1 = 7.00 + 0.5 = 7.50
t=2: X_2 = 0.7*7.5 + ε_2 = 5.25 - 0.3 = 4.95
t=3: X_3 = 0.7*4.95 + ε_3 = 3.47 + 0.8 = 4.27

Reversión a la media hacia 0 con tasa de decaimiento 0.7
```

**Aplicación en Trading**:
- Estrategias de reversión a la media para pares de trading
- Predicción de movimientos de precios a corto plazo
- Identificación de condiciones de sobrecompra/sobreventa

**Riesgos**:
- Asume relaciones lineales (los mercados son a menudo no lineales)
- La suposición de estacionariedad a menudo se viola en mercados reales
- Cambios estructurales invalidan el modelo

---

### 1.2 Procesos de Medias Móviles

**Intuición**: Los procesos MA modelan el valor actual como una suma ponderada de shocks/sorpresas recientes, capturando impactos temporales.

**Explicación Geométrica**:
- Piensa en ondas en el agua después de tirar piedras
- Cada "shock" crea ondas que se desvanecen con el tiempo
- MA(q) = q shocks previos todavía afectando el valor actual

**Definición Matemática**:

Proceso MA(q):
$$
X_t = \mu + \varepsilon_t + \sum_{i=1}^{q} \theta_i \varepsilon_{t-i}
$$

donde:
- $\mu$ es la media
- $\theta_i$ son coeficientes MA
- $\varepsilon_t \sim N(0, \sigma^2)$ son shocks independientes

**Condición de Invertibilidad**:
$$
1 + \theta_1 z + \theta_2 z^2 + ... + \theta_q z^q = 0 \implies |z| > 1
$$

**Ejemplo Numérico - MA(2)**:
```
Dado: X_t = 5 + ε_t + 0.6*ε_{t-1} + 0.3*ε_{t-2}

Shocks: ε_0=1, ε_1=-0.5, ε_2=0.8, ε_3=0.2

t=0: X_0 = 5 + 1.0 + 0 + 0 = 6.00
t=1: X_1 = 5 + (-0.5) + 0.6*1.0 + 0 = 5.10
t=2: X_2 = 5 + 0.8 + 0.6*(-0.5) + 0.3*1.0 = 5.80
t=3: X_3 = 5 + 0.2 + 0.6*0.8 + 0.3*(-0.5) = 5.53
```

---

## Procesos Estocásticos

### 2.1 Movimiento Browniano (Proceso de Wiener)

**Intuición**: Caminata aleatoria en tiempo continuo - como el camino de una persona ebria, cada paso es aleatorio.

**Visualización Geométrica**:
```
Trayectoria del precio:
    ^
    |     /\  /\/\
    |    /  \/    \
    |   /          \/\
    |  /              \
    +-------------------->
      Tiempo
```

**Propiedades Matemáticas**:

Un proceso $W_t$ es movimiento Browniano si:
1. $W_0 = 0$
2. $W_t$ tiene incrementos independientes
3. $W_t - W_s \sim N(0, t-s)$ para $t > s$
4. $W_t$ tiene trayectorias continuas

**Variación Cuadrática**:
$$
[W, W]_t = t
$$

Esta propiedad fundamental conduce al lema de Itô.

**Ejemplo Numérico**:
```python
# Simulando Movimiento Browniano
dt = 0.01
n_steps = 1000
dW = np.random.normal(0, np.sqrt(dt), n_steps)
W = np.cumsum(dW)

# Propiedades:
# Var(W_1) ≈ 1.0
# E[W_1] ≈ 0
```

**Aplicación en Trading**:
- Base del modelo Black-Scholes para valoración de opciones
- Modelado de movimientos aleatorios de precios
- Simulaciones Monte Carlo

---

### 2.2 Movimiento Browniano Geométrico (MBG)

**Intuición**: Los precios de acciones siguen MBG - no pueden ser negativos, y los cambios porcentuales están normalmente distribuidos.

**Definición Matemática**:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

**Solución** (vía lema de Itô):
$$
S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

**Ejemplo Numérico**:
```
S_0 = 100, μ = 0.10 (10% deriva), σ = 0.20 (20% vol), dt = 1/252

Día 1: dW = 0.05
S_1 = 100 * exp((0.10 - 0.20²/2)*1/252 + 0.20*0.05)
    = 100 * exp(0.000357 + 0.01)
    = 100 * 1.0104 = 101.04

Día 2: dW = -0.08
S_2 = 101.04 * exp(0.000357 - 0.016)
    = 101.04 * 0.9844 = 99.46
```

**Realidad del Trading**:
- Los mercados reales tienen colas gordas (movimientos más grandes de lo que MBG predice)
- La volatilidad se agrupa (efectos GARCH)
- Componentes de salto (caídas repentinas)

---

## Modelado de Volatilidad

### 3.1 Proceso ARCH

**Intuición**: La volatilidad de hoy depende de los retornos al cuadrado de ayer - los grandes movimientos se agrupan.

**Explicación Geométrica**:
- La volatilidad viene en olas
- Períodos de calma → Períodos de caos → Períodos de calma
- "Agrupamiento de volatilidad" - observación de Mandelbrot

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

**Restricciones**:
- $\omega > 0$
- $\alpha_i \geq 0$ para todo $i$
- $\sum_{i=1}^{q} \alpha_i < 1$ (estacionariedad)

**Ejemplo Numérico - ARCH(1)**:
```
Parámetros: ω = 0.01, α = 0.3

t=0: r_0 = 2%, σ_0² = 0.04%
t=1: σ_1² = 0.01 + 0.3*(2%)² = 0.01 + 0.00012 = 0.01012
     σ_1 = 10.06%, ε_1 = -0.5
     r_1 = 10.06% * (-0.5) = -5.03%
     
t=2: σ_2² = 0.01 + 0.3*(-5.03%)² = 0.01 + 0.000759 = 0.010759
     σ_2 = 10.37%
```

**Aplicación en Trading**:
- Pronóstico de volatilidad para valoración de opciones
- Gestión de riesgos (cálculos de VaR)
- Dimensionamiento de posiciones basado en regímenes de volatilidad

**Errores Comunes**:
- Usar ARCH para pronósticos a largo plazo (revierte rápidamente a la media)
- Ignorar efectos de apalancamiento (usar EGARCH en su lugar)
- No verificar residuos para autocorrelación restante

---

### 3.2 Proceso GARCH

**Intuición**: GARCH añade "momentum" a la volatilidad - recuerda tanto shocks pasados COMO niveles de volatilidad pasados.

**Definición Matemática**:

Modelo GARCH(p,q):
$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i r_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$

**Más Común**: GARCH(1,1)
$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

**Varianza Incondicional**:
$$
\text{Var}(r_t) = \frac{\omega}{1 - \alpha - \beta}
$$

**Persistencia**: $\alpha + \beta$ mide la persistencia de la volatilidad
- Cerca de 1: muy persistente (decaimiento lento)
- Lejos de 1: reversión rápida a la media

**Ejemplo Numérico - GARCH(1,1)**:
```
Parámetros: ω = 0.00001, α = 0.08, β = 0.90
(Nota: α + β = 0.98 → altamente persistente)

Inicial: σ_0² = 0.0004 (vol diaria del 2%)

t=1: r_1 = 3%
     σ_1² = 0.00001 + 0.08*(0.03)² + 0.90*0.0004
          = 0.00001 + 0.000072 + 0.00036
          = 0.000442
     σ_1 = 2.10%

t=2: r_2 = -1%
     σ_2² = 0.00001 + 0.08*(-0.01)² + 0.90*0.000442
          = 0.00001 + 0.000008 + 0.000398
          = 0.000416
     σ_2 = 2.04%
```

**Aplicación Paso a Paso al Trading**:

1. **Estimar GARCH con datos históricos** (ej. 1 año de retornos diarios)
2. **Pronosticar volatilidad de mañana**: Usar la ecuación GARCH
3. **Ajustar tamaño de posición**: 
   - Si $\sigma_{t+1}$ alto → Reducir tamaño de posición
   - Si $\sigma_{t+1}$ bajo → Aumentar tamaño de posición
4. **Valoración de opciones**: Usar volatilidad pronosticada en Black-Scholes
5. **Colocación de stop-loss**: Establecer stops en $k \times \sigma_{t+1}$

**Combinando con RSI**:
```python
# Lógica de Trading
if RSI < 30 and σ_forecast < σ_avg:
    # Sobrevendido + Baja volatilidad → Alta convicción de compra
    position_size = 2.0
elif RSI < 30 and σ_forecast > σ_avg:
    # Sobrevendido + Alta volatilidad → Compra cautelosa
    position_size = 0.5
```

---

### 3.3 EGARCH (GARCH Exponencial)

**Intuición**: Las malas noticias (retornos negativos) aumentan la volatilidad MÁS que las buenas noticias de la misma magnitud.

**¿Por qué EGARCH?**
- GARCH es simétrico: +5% y -5% tienen el mismo impacto en la volatilidad
- Realidad: una caída del -5% causa más miedo/volatilidad que una ganancia del +5%
- EGARCH captura el "efecto de apalancamiento"

**Definición Matemática**:
$$
\log(\sigma_t^2) = \omega + \sum_{i=1}^{q} \left[\alpha_i \left|\frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right| + \gamma_i \frac{\varepsilon_{t-i}}{\sigma_{t-i}}\right] + \sum_{j=1}^{p} \beta_j \log(\sigma_{t-j}^2)
$$

**EGARCH(1,1)**:
$$
\log(\sigma_t^2) = \omega + \alpha \left|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}\right| + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}} + \beta \log(\sigma_{t-1}^2)
$$

**Parámetro Clave**: $\gamma$ (efecto de apalancamiento)
- $\gamma < 0$: Shocks negativos aumentan más la volatilidad (típico)
- $\gamma = 0$: Simétrico (como GARCH)
- $\gamma > 0$: Shocks positivos aumentan más la volatilidad (raro)

**Ejemplo Numérico**:
```
Parámetros: ω = -0.1, α = 0.15, γ = -0.05, β = 0.95

Inicial: log(σ_0²) = -6.0 → σ_0 = 5%

Escenario A: Shock positivo (ε_1/σ_0 = +1.5)
log(σ_1²) = -0.1 + 0.15*|1.5| + (-0.05)*1.5 + 0.95*(-6.0)
          = -0.1 + 0.225 - 0.075 - 5.7
          = -5.65
σ_1 = 5.82%

Escenario B: Shock negativo (ε_1/σ_0 = -1.5)
log(σ_1²) = -0.1 + 0.15*|-1.5| + (-0.05)*(-1.5) + 0.95*(-6.0)
          = -0.1 + 0.225 + 0.075 - 5.7
          = -5.50
σ_1 = 6.74%

→ ¡El shock negativo aumenta más la volatilidad! (6.74% vs 5.82%)
```

**Trading con EGARCH**:

1. **Detección de Caídas**: Si γ < 0 y retorno grande negativo → Esperar mayor volatilidad
2. **Gestión de Riesgos**: Ajustar stops después de movimientos negativos
3. **Estrategias de Opciones**: 
   - Comprar opciones put después de caídas del mercado (la vol aumentará)
   - Vender opciones call después de rallies (la vol no aumentará tanto)

---

## Modelos Ocultos de Markov

### 4.1 Conceptos Fundamentales

**Intuición**: Los mercados tienen "estados" ocultos (alcista, bajista, lateral) que no podemos observar directamente, pero podemos inferir del comportamiento de precios.

**Explicación Geométrica**:
```
Estados Ocultos:  [Alcista] → [Alcista] → [Bajista] → [Bajista] → [Alcista]
                      ↓           ↓           ↓           ↓           ↓
Observable:        [+2%]       [+1%]       [-3%]       [-2%]       [+1%]
```

**Componentes Matemáticos**:

1. **Estados**: $S = \{s_1, s_2, ..., s_N\}$
2. **Observaciones**: $O = \{o_1, o_2, ..., o_T\}$
3. **Matriz de Transición**: $A = [a_{ij}]$ donde $a_{ij} = P(s_t = j | s_{t-1} = i)$
4. **Matriz de Emisión**: $B = [b_j(k)]$ donde $b_j(k) = P(o_t = k | s_t = j)$
5. **Distribución Inicial**: $\pi = [\pi_i]$ donde $\pi_i = P(s_1 = i)$

**Ejemplo: Modelo de Mercado de 2 Estados**

Estados: {Alcista, Bajista}

Matriz de Transición A:
```
            Alcista  Bajista
Alcista  [   0.90    0.10  ]
Bajista  [   0.20    0.80  ]
```
Interpretación: 
- Si está Alcista, 90% de probabilidad de mantenerse Alcista, 10% de cambiar a Bajista
- Si está Bajista, 80% de probabilidad de mantenerse Bajista, 20% de cambiar a Alcista

Distribuciones de Emisión (Gaussianas):
```
Alcista: μ = 0.05%, σ = 1.0%  (pequeña deriva positiva, baja vol)
Bajista: μ = -0.10%, σ = 2.5% (deriva negativa, alta vol)
```

**Ejemplo Numérico**:

Observaciones: [+0.5%, +0.3%, -2.0%, -1.5%, +0.2%]

Paso 1: Inicializar (algoritmo forward)
```
π(Alcista) = 0.6, π(Bajista) = 0.4

t=1, o_1 = +0.5%:
P(o_1|Alcista) = N(0.5; 0.05, 1.0) = 0.398
P(o_1|Bajista) = N(0.5; -0.10, 2.5) = 0.155

α_1(Alcista) = π(Alcista) * P(o_1|Alcista) = 0.6 * 0.398 = 0.239
α_1(Bajista) = π(Bajista) * P(o_1|Bajista) = 0.4 * 0.155 = 0.062
```

Paso 2: Recursión
```
t=2, o_2 = +0.3%:
α_2(Alcista) = P(o_2|Alcista) * [α_1(Alcista)*0.90 + α_1(Bajista)*0.20]
            = 0.396 * [0.239*0.90 + 0.062*0.20]
            = 0.396 * 0.227 = 0.090
         
α_2(Bajista) = P(o_2|Bajista) * [α_1(Alcista)*0.10 + α_1(Bajista)*0.80]
            = 0.161 * [0.239*0.10 + 0.062*0.80]
            = 0.161 * 0.074 = 0.012
```

**Algoritmo de Viterbi** (Secuencia de Estados Más Probable):

Encontrar: $\arg\max_{s_1,...,s_T} P(s_1,...,s_T | o_1,...,o_T)$

```python
# Pseudocódigo
δ[t][i] = probabilidad máxima del estado i en el tiempo t
ψ[t][i] = mejor estado previo

# Inicialización
δ[1][i] = π[i] * B[i][o_1]

# Recursión
for t in range(2, T+1):
    for j in range(N):
        δ[t][j] = max_i(δ[t-1][i] * A[i][j]) * B[j][o_t]
        ψ[t][j] = argmax_i(δ[t-1][i] * A[i][j])

# Retroceso
s_T = argmax_i(δ[T][i])
for t in range(T-1, 0, -1):
    s_t = ψ[t+1][s_{t+1}]
```

**Aplicación de Trading con HMM**:

```python
# Estrategia de Detección de Régimen
if estado_actual == 'Alcista':
    # Seguimiento de tendencia
    if close > MA20:
        señal = 'COMPRAR'
    # Pero prepararse para cambio de régimen
    if P(transición a Bajista) > 0.3:
        reducir_tamaño_posicion()

elif estado_actual == 'Bajista':
    # Reversión a la media o quedarse fuera
    if RSI < 20:
        señal = 'COMPRAR'  # Sobrevendido en mercado bajista
    else:
        señal = 'MANTENER' or 'VENDER'
        
elif estado_actual == 'Lateral':
    # Trading de rango
    if close < soporte + tolerancia:
        señal = 'COMPRAR'
    elif close > resistencia - tolerancia:
        señal = 'VENDER'
```

**Errores Comunes**:
1. **Demasiados estados**: Comenzar con 2-3 estados, no 10
2. **Sobreajuste**: HMM puede ajustar ruido si es demasiado complejo
3. **Ignorar incertidumbre**: Usar probabilidades de estado, no solo el estado más probable
4. **Sesgo de anticipación**: No entrenar con datos futuros

---

## Teoría de Optimización

### 5.1 Optimización Convexa

**Intuición**: Encontrar la mejor solución donde "mejor" no es ambiguo - hay un mínimo global único.

**Explicación Geométrica**:
```
No convexa (mala):           Convexa (buena):
    ^                           ^
    | /\  /\                   |    /\
    |/  \/  \                  |   /  \
    |        \                 |  /    \
    +---------->               +----------->
Múltiples mínimos           Mínimo único
```

**Definición Matemática**:

Una función $f: \mathbb{R}^n \to \mathbb{R}$ es convexa si:
$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)
$$
para todo $x, y \in \mathbb{R}^n$ y $\lambda \in [0, 1]$.

**Propiedad Clave**: ¡Cualquier mínimo local es un mínimo global!

**Ejemplos en Finanzas**:

1. **Optimización Media-Varianza** (Markowitz):
$$
\min_w \quad w^T \Sigma w
$$
$$
\text{s.t.} \quad w^T \mu \geq r_{\text{objetivo}}, \quad w^T \mathbf{1} = 1
$$

Esto es convexo porque:
- Objetivo: $w^T \Sigma w$ es cuadrático con $\Sigma \succeq 0$
- Restricciones: Lineales

2. **Portafolio con Costos de Transacción**:
$$
\min_w \quad w^T \Sigma w + \kappa ||w - w_{\text{anterior}}||_1
$$

¡Todavía convexo! (la norma L1 es convexa)

**Ejemplo Numérico - Descenso de Gradiente**:
```
Objetivo: f(w) = w² - 4w + 4 = (w-2)²
Gradiente: f'(w) = 2w - 4

Punto inicial: w_0 = 0
Tasa de aprendizaje: α = 0.1

Iteración 1:
  f'(0) = -4
  w_1 = 0 - 0.1*(-4) = 0.4
  
Iteración 2:
  f'(0.4) = 2*0.4 - 4 = -3.2
  w_2 = 0.4 - 0.1*(-3.2) = 0.72
  
Iteración 3:
  f'(0.72) = 2*0.72 - 4 = -2.56
  w_3 = 0.72 - 0.1*(-2.56) = 0.976
  
Converge a w* = 2.0
```

---

### 5.2 Lagrangiano y Condiciones KKT

**Intuición**: Convertir problema con restricciones en no restringido "penalizando" violaciones de restricciones.

**El Lagrangiano**:
$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)
$$

donde:
- $f(x)$: función objetivo
- $g_i(x) \leq 0$: restricciones de desigualdad
- $h_j(x) = 0$: restricciones de igualdad
- $\lambda_i, \nu_j$: multiplicadores de Lagrange

**Condiciones KKT** (necesarias para optimalidad):
1. **Estacionariedad**: $\nabla_x \mathcal{L} = 0$
2. **Factibilidad primal**: $g_i(x) \leq 0$, $h_j(x) = 0$
3. **Factibilidad dual**: $\lambda_i \geq 0$
4. **Holgura complementaria**: $\lambda_i g_i(x) = 0$

**Ejemplo: Optimización de Portafolio**

$$
\min_w \quad \frac{1}{2} w^T \Sigma w
$$
$$
\text{s.t.} \quad w^T \mu = r_0, \quad w^T \mathbf{1} = 1
$$

Lagrangiano:
$$
\mathcal{L}(w, \lambda_1, \lambda_2) = \frac{1}{2} w^T \Sigma w + \lambda_1(w^T \mu - r_0) + \lambda_2(w^T \mathbf{1} - 1)
$$

Condiciones KKT:
$$
\Sigma w + \lambda_1 \mu + \lambda_2 \mathbf{1} = 0
$$
$$
w^T \mu = r_0, \quad w^T \mathbf{1} = 1
$$

**Solución**:
$$
w^* = \Sigma^{-1} \left[\lambda_1 \mu + \lambda_2 \mathbf{1}\right]
$$

Resolver para $\lambda_1, \lambda_2$ usando las restricciones.

---

## Teoría de la Información

### 6.1 Entropía e Información

**Intuición**: La entropía mide sorpresa/incertidumbre. Alta entropía = difícil de predecir.

**Definición Matemática**:

Entropía de Shannon:
$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
$$

**Ejemplo Numérico**:
```
Moneda A: P(Cara) = 0.5, P(Cruz) = 0.5
H(A) = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1 bit

Moneda B: P(Cara) = 0.9, P(Cruz) = 0.1
H(B) = -0.9*log₂(0.9) - 0.1*log₂(0.1) = 0.47 bits

¡La moneda A tiene más incertidumbre!
```

**Aplicación en Trading**:

1. **Eficiencia del Mercado**: Alta entropía → Difícil de predecir → Mercado eficiente
2. **Decaimiento del Alpha**: A medida que la entropía disminuye, tu ventaja decae
3. **Ratio de Información**: Mide ratio señal/ruido

**Pérdida de Entropía Cruzada** (para modelos ML):
$$
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

Usada en clasificación: predecir arriba/abajo/lateral

---

### 6.2 Información Mutua

**Definición**:
$$
I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

**Intuición**: ¿Cuánto nos dice conocer X sobre Y?

**Ejemplo de Trading**:
```
X = {RSI sobrecomprado, RSI normal}
Y = {Precio baja al día siguiente, Precio sube al día siguiente}

Alto I(X;Y) → RSI es informativo
Bajo I(X;Y) → RSI no proporciona ventaja
```

**Calculando IM para Señales de Trading**:
```python
def mutual_information(señal, retornos):
    """
    Discretizar señal y retornos
    Calcular distribuciones conjuntas y marginales
    Calcular IM
    """
    # Ejemplo:
    # Si IM > 0.1 bits → Señal es útil
    # Si IM < 0.01 bits → Señal es ruido
```

---

## Resumen: Aplicando al Trading Algorítmico

### Marco de Integración

**1. Pipeline de Datos**:
```
Precios raw → Retornos → Ingeniería de features → Detección de régimen (HMM)
```

**2. Pronóstico de Volatilidad**:
```
Retornos → GARCH/EGARCH → σ_pronóstico → Dimensionamiento de posición
```

**3. Generación de Señales**:
```
Indicadores técnicos (RSI, Divergencias) → Modelo ML → Señal de trading
```

**4. Construcción de Portafolio**:
```
Señales + Pronósticos de volatilidad → Optimización (media-varianza) → Pesos
```

**5. Gestión de Riesgos**:
```
Posiciones → VaR (de GARCH) → Niveles de stop-loss
```

### Combinando Todo

**Sistema de Trading de Ejemplo**:

```python
# Paso 1: Detectar régimen
régimen = hmm.predict(retornos_recientes)

# Paso 2: Pronosticar volatilidad
pronóstico_vol = garch.forecast(horizon=1)

# Paso 3: Generar señales
rsi = calcular_rsi(precios)
divergencia = detectar_divergencia(precios, rsi)
señal_ml = modelo_lstm.predict(features)

# Paso 4: Combinar señales basadas en régimen
if régimen == 'Alcista' and rsi < 30 and divergencia == 'alcista':
    señal = señal_ml * 1.5  # Alta convicción
elif régimen == 'Bajista':
    señal = 0  # Quedarse fuera
else:
    señal = señal_ml * 0.5  # Baja convicción

# Paso 5: Dimensionamiento de posición
posición = señal / pronóstico_vol  # Ponderación inversa de volatilidad

# Paso 6: Gestión de riesgos
stop_loss = precio_entrada - 2 * pronóstico_vol
```

---

## Referencias

1. Tsay, R. S. (2010). *Análisis de Series Temporales Financieras*. Wiley.
2. Hamilton, J. D. (1994). *Análisis de Series Temporales*. Princeton University Press.
3. Murphy, K. P. (2012). *Aprendizaje Automático: Una Perspectiva Probabilística*. MIT Press.
4. Boyd, S., & Vandenberghe, L. (2004). *Optimización Convexa*. Cambridge University Press.
5. Cover, T. M., & Thomas, J. A. (2006). *Elementos de Teoría de la Información*. Wiley.

---

**Próximos Pasos**: Proceder a los notebooks específicos de módulos para implementaciones prácticas y ejercicios.
