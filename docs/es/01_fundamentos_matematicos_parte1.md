# Fundamentos Matemáticos para ML Cuantitativo - Parte 1

## Series Temporales y Procesos Estocásticos

### 1.1 Procesos Autoregresivos (AR)

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

**Riesgos Comunes**:
- Asume relaciones lineales (los mercados son a menudo no lineales)
- La suposición de estacionariedad a menudo se viola en mercados reales
- Cambios estructurales invalidan el modelo

---

### 1.2 Procesos de Medias Móviles (MA)

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

**Aplicación en Trading**:
- Modelado de impactos temporales de noticias
- Suavizado de señales de trading
- Análisis de respuestas a eventos del mercado

---

### 1.3 Movimiento Browniano (Proceso de Wiener)

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
- Simulaciones Monte Carlo para gestión de riesgos

---

### 1.4 Movimiento Browniano Geométrico (MBG)

**Intuición**: Los precios de acciones siguen MBG - no pueden ser negativos, y los cambios porcentuales están normalmente distribuidos.

**Definición Matemática**:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

**Solución** (vía lema de Itô):
$$
S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

**Parámetros**:
- $\mu$: deriva (retorno esperado)
- $\sigma$: volatilidad (desviación estándar de retornos)
- $S_0$: precio inicial

**Ejemplo Numérico**:
```
S_0 = 100, μ = 0.10 (10% deriva anual), σ = 0.20 (20% vol anual), dt = 1/252

Día 1: dW = 0.05
S_1 = 100 * exp((0.10 - 0.20²/2)*1/252 + 0.20*0.05)
    = 100 * exp(0.000357 + 0.01)
    = 100 * 1.0104 = 101.04

Día 2: dW = -0.08
S_2 = 101.04 * exp(0.000357 - 0.016)
    = 101.04 * 0.9844 = 99.46
```

**Realidad vs MBG en Trading**:
- ✅ Los precios son positivos
- ✅ Captura la deriva y volatilidad
- ❌ Los mercados reales tienen colas gordas (más movimientos extremos)
- ❌ La volatilidad se agrupa (efectos GARCH)
- ❌ No captura saltos (crashes repentinos)

**Mejoras del Modelo**:
- Agregar componentes de salto (modelo Merton)
- Volatilidad estocástica (modelo Heston)
- Modelado GARCH para volatilidad variable

---

## Resumen de Parte 1

Has aprendido los bloques fundamentales de las series temporales:
- **AR**: El pasado predice el futuro
- **MA**: Los shocks tienen efectos temporales
- **Brownian Motion**: Aleatoriedad en tiempo continuo
- **GBM**: Modelo realista para precios de acciones

**Próxima Parte**: Modelado de volatilidad (ARCH, GARCH, EGARCH)
