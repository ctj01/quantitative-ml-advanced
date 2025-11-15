# Fundamentos Matemáticos para ML Cuantitativo - Parte 3

## Modelos Ocultos de Markov (HMM)

### 3.1 Conceptos Fundamentales

**Intuición**: Los mercados tienen "estados" ocultos (alcista, bajista, lateral) que no podemos observar directamente, pero podemos inferir del comportamiento de precios.

**Explicación Geométrica**:
```
Estados Ocultos:  [Alcista] → [Alcista] → [Bajista] → [Bajista] → [Alcista]
                      ↓           ↓           ↓           ↓           ↓
Observable:        [+2%]       [+1%]       [-3%]       [-2%]       [+1%]

No vemos directamente el estado, solo las observaciones (retornos)
```

**Componentes Matemáticos**:

Un HMM está definido por:

1. **Estados**: $S = \{s_1, s_2, ..., s_N\}$
   - Ejemplo: {Bajista, Lateral, Alcista}

2. **Observaciones**: $O = \{o_1, o_2, ..., o_T\}$
   - Ejemplo: Retornos diarios

3. **Matriz de Transición**: $A = [a_{ij}]$
   $$a_{ij} = P(s_t = j | s_{t-1} = i)$$
   - Probabilidad de ir del estado $i$ al estado $j$

4. **Matriz de Emisión**: $B = [b_j(k)]$
   $$b_j(k) = P(o_t = k | s_t = j)$$
   - Probabilidad de observar $k$ dado que estamos en estado $j$

5. **Distribución Inicial**: $\pi = [\pi_i]$
   $$\pi_i = P(s_1 = i)$$

**Notación Compacta**: $\lambda = (A, B, \pi)$

---

### 3.2 Ejemplo de Mercado con 2 Estados

**Modelo Simple**: Alcista vs Bajista

**Matriz de Transición A**:
```
            Alcista  Bajista
Alcista  [   0.90    0.10  ]
Bajista  [   0.20    0.80  ]
```

Interpretación:
- Si hoy es Alcista → 90% de probabilidad de seguir Alcista mañana
- Si hoy es Alcista → 10% de probabilidad de cambiar a Bajista
- Si hoy es Bajista → 80% de probabilidad de seguir Bajista mañana
- Si hoy es Bajista → 20% de probabilidad de cambiar a Alcista

**Distribuciones de Emisión** (Gaussianas):
```
Estado Alcista:
- μ_bull = +0.08% (deriva positiva diaria)
- σ_bull = 1.0% (baja volatilidad)

Estado Bajista:
- μ_bear = -0.15% (deriva negativa diaria)
- σ_bear = 2.5% (alta volatilidad)
```

**Ejemplo Numérico Completo**:

```
Observaciones: retornos = [+0.5%, +0.3%, -2.0%, -1.5%, +0.2%]

Queremos: ¿Cuál era el estado en cada día?

Paso 1: Inicialización (algoritmo forward)
-------
Distribución inicial: π(Alcista) = 0.6, π(Bajista) = 0.4

Día 1, retorno = +0.5%:
  P(r=+0.5% | Alcista) = N(0.5; μ=0.08, σ=1.0) = 0.398
  P(r=+0.5% | Bajista) = N(0.5; μ=-0.15, σ=2.5) = 0.155

  α_1(Alcista) = π(Alcista) × P(r|Alcista) = 0.6 × 0.398 = 0.239
  α_1(Bajista) = π(Bajista) × P(r|Bajista) = 0.4 × 0.155 = 0.062

Paso 2: Recursión
-------
Día 2, retorno = +0.3%:
  P(r=+0.3% | Alcista) = N(0.3; 0.08, 1.0) = 0.396
  P(r=+0.3% | Bajista) = N(0.3; -0.15, 2.5) = 0.161

  α_2(Alcista) = P(r|Alcista) × [α_1(Alcista)×0.90 + α_1(Bajista)×0.20]
               = 0.396 × [0.239×0.90 + 0.062×0.20]
               = 0.396 × 0.227
               = 0.090
         
  α_2(Bajista) = P(r|Bajista) × [α_1(Alcista)×0.10 + α_1(Bajista)×0.80]
               = 0.161 × [0.239×0.10 + 0.062×0.80]
               = 0.161 × 0.074
               = 0.012

Día 3, retorno = -2.0% (shock negativo):
  P(r=-2.0% | Alcista) = N(-2.0; 0.08, 1.0) = 0.012 (muy bajo!)
  P(r=-2.0% | Bajista) = N(-2.0; -0.15, 2.5) = 0.156 (más probable)

  α_3(Alcista) = 0.012 × [0.090×0.90 + 0.012×0.20] = 0.012 × 0.083 = 0.001
  α_3(Bajista) = 0.156 × [0.090×0.10 + 0.012×0.80] = 0.156 × 0.019 = 0.003

Estado más probable en día 3: BAJISTA (0.003 > 0.001)
```

---

### 3.3 Algoritmo de Viterbi

**Objetivo**: Encontrar la secuencia de estados más probable

$$\text{Maximizar: } P(s_1, s_2, ..., s_T | o_1, o_2, ..., o_T)$$

**Programación Dinámica**:

```python
def viterbi(observaciones, A, B, pi):
    """
    Algoritmo de Viterbi para encontrar secuencia óptima de estados
    
    Args:
        observaciones: Lista de observaciones [o_1, o_2, ..., o_T]
        A: Matriz de transición (N×N)
        B: Matriz de emisión (N×M) o funciones de densidad
        pi: Distribución inicial (N,)
    
    Returns:
        estados: Secuencia óptima de estados
        prob_max: Probabilidad de esa secuencia
    """
    N = len(pi)  # Número de estados
    T = len(observaciones)  # Número de observaciones
    
    # Matrices de programación dinámica
    delta = np.zeros((T, N))  # Probabilidad máxima
    psi = np.zeros((T, N), dtype=int)  # Mejor estado previo
    
    # Inicialización (t=0)
    for i in range(N):
        delta[0, i] = pi[i] * B[i](observaciones[0])
        psi[0, i] = 0
    
    # Recursión (t=1 hasta T-1)
    for t in range(1, T):
        for j in range(N):
            # Para estado j en tiempo t
            # Encontrar mejor estado previo i
            probabilidades = delta[t-1, :] * A[:, j]
            mejor_i = np.argmax(probabilidades)
            
            delta[t, j] = probabilidades[mejor_i] * B[j](observaciones[t])
            psi[t, j] = mejor_i
    
    # Terminación: Encontrar mejor último estado
    estados = np.zeros(T, dtype=int)
    estados[T-1] = np.argmax(delta[T-1, :])
    prob_max = delta[T-1, estados[T-1]]
    
    # Backtracking: Reconstruir secuencia óptima
    for t in range(T-2, -1, -1):
        estados[t] = psi[t+1, estados[t+1]]
    
    return estados, prob_max
```

**Ejemplo de Uso**:

```python
import numpy as np
from scipy.stats import norm

# Definir modelo HMM
A = np.array([[0.90, 0.10],   # Alcista → [Alcista, Bajista]
              [0.20, 0.80]])   # Bajista → [Alcista, Bajista]

# Distribuciones de emisión (Gaussianas)
def B_alcista(r):
    return norm.pdf(r, loc=0.08, scale=1.0)

def B_bajista(r):
    return norm.pdf(r, loc=-0.15, scale=2.5)

B = [B_alcista, B_bajista]

pi = np.array([0.6, 0.4])  # 60% alcista, 40% bajista inicialmente

# Observaciones (retornos en %)
observaciones = np.array([0.5, 0.3, -2.0, -1.5, 0.2])

# Ejecutar Viterbi
estados, prob = viterbi(observaciones, A, B, pi)

print("Secuencia de estados óptima:")
nombres = ['Alcista', 'Bajista']
for t, estado in enumerate(estados):
    print(f"Día {t+1}: {nombres[estado]} (retorno={observaciones[t]:.1f}%)")

# Output:
# Día 1: Alcista (retorno=0.5%)
# Día 2: Alcista (retorno=0.3%)
# Día 3: Bajista (retorno=-2.0%)
# Día 4: Bajista (retorno=-1.5%)
# Día 5: Alcista (retorno=0.2%)
```

---

### 3.4 Estimación de Parámetros: Algoritmo Baum-Welch

**Problema**: Tenemos observaciones pero no sabemos A, B, π

**Solución**: EM (Expectation-Maximization) - Algoritmo Baum-Welch

**Algoritmo**:

```python
from hmmlearn import hmm

def entrenar_hmm(retornos, n_estados=3, n_iter=100):
    """
    Entrenar HMM con Baum-Welch
    
    Args:
        retornos: Array de retornos históricos
        n_estados: Número de estados ocultos
        n_iter: Iteraciones máximas
    
    Returns:
        modelo: HMM entrenado
    """
    # Preparar datos (reshape para hmmlearn)
    X = retornos.reshape(-1, 1)
    
    # Inicializar modelo
    modelo = hmm.GaussianHMM(
        n_components=n_estados,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42
    )
    
    # Entrenar
    modelo.fit(X)
    
    # Extraer parámetros aprendidos
    print("Matriz de Transición:")
    print(modelo.transmat_)
    print("\nMedias de Estados:")
    print(modelo.means_)
    print("\nCovarianzas de Estados:")
    print(modelo.covars_)
    
    return modelo

# Ejemplo con datos reales
retornos = df['close'].pct_change().dropna().values * 100  # En %
modelo_hmm = entrenar_hmm(retornos, n_estados=3)

# Predecir estados
estados_predichos = modelo_hmm.predict(retornos.reshape(-1, 1))

# Analizar estados
for estado in range(3):
    retornos_estado = retornos[estados_predichos == estado]
    print(f"\nEstado {estado}:")
    print(f"  Media: {retornos_estado.mean():.3f}%")
    print(f"  Std: {retornos_estado.std():.3f}%")
    print(f"  Días: {len(retornos_estado)}")

# Típicamente verás algo como:
# Estado 0: Media=-0.15%, Std=2.5% → BAJISTA
# Estado 1: Media=0.01%, Std=0.8% → LATERAL
# Estado 2: Media=0.12%, Std=1.2% → ALCISTA
```

---

### 3.5 Aplicación en Trading

**Estrategia de Cambio de Régimen**:

```python
class EstrategiaHMMRegimen:
    def __init__(self, n_estados=3):
        self.modelo_hmm = None
        self.n_estados = n_estados
        self.estados_nombres = ['Bajista', 'Lateral', 'Alcista']
        
    def entrenar(self, df_historico):
        """Entrenar HMM con datos históricos"""
        retornos = df_historico['close'].pct_change().dropna().values
        retornos = retornos.reshape(-1, 1)
        
        self.modelo_hmm = hmm.GaussianHMM(
            n_components=self.n_estados,
            covariance_type="full",
            n_iter=100
        )
        self.modelo_hmm.fit(retornos)
        
        # Ordenar estados por media (Bajista < Lateral < Alcista)
        medias = self.modelo_hmm.means_.flatten()
        orden = np.argsort(medias)
        self.mapeo_estados = {i: orden[i] for i in range(self.n_estados)}
        
    def predecir_estado_actual(self, df_reciente):
        """Predecir estado actual del mercado"""
        retornos = df_reciente['close'].pct_change().dropna().values
        retornos = retornos.reshape(-1, 1)
        
        # Predecir secuencia
        estados = self.modelo_hmm.predict(retornos)
        estado_actual_raw = estados[-1]
        
        # Mapear a nombre
        estado_actual = self.mapeo_estados[estado_actual_raw]
        
        # Calcular probabilidad de transición
        prob_transicion = self.modelo_hmm.transmat_[estado_actual_raw, :]
        
        return {
            'estado': self.estados_nombres[estado_actual],
            'confianza': prob_transicion[estado_actual_raw],
            'prob_cambio': 1 - prob_transicion[estado_actual_raw]
        }
    
    def generar_señal(self, df, indicadores):
        """
        Generar señal basada en estado + indicadores técnicos
        """
        info_estado = self.predecir_estado_actual(df)
        estado = info_estado['estado']
        rsi = indicadores['rsi']
        
        señal = None
        
        if estado == 'Alcista':
            # En alcista: seguir tendencia
            if rsi < 50 and info_estado['confianza'] > 0.7:
                señal = {
                    'tipo': 'COMPRAR',
                    'razón': f'Régimen {estado}, RSI={rsi:.1f}',
                    'confianza': info_estado['confianza'],
                    'tamaño': 0.08  # 8% del capital
                }
        
        elif estado == 'Bajista':
            # En bajista: esperar reversión extrema
            if rsi < 25 and info_estado['prob_cambio'] > 0.25:
                señal = {
                    'tipo': 'COMPRAR',
                    'razón': f'Régimen {estado} pero RSI extremo, posible cambio',
                    'confianza': info_estado['prob_cambio'],
                    'tamaño': 0.03  # 3% del capital (conservador)
                }
            elif rsi > 50:
                señal = {
                    'tipo': 'VENDER',
                    'razón': f'Régimen {estado}, vender rebotes',
                    'confianza': info_estado['confianza'],
                    'tamaño': 0.05
                }
        
        elif estado == 'Lateral':
            # En lateral: reversión a la media
            if rsi < 30:
                señal = {
                    'tipo': 'COMPRAR',
                    'razón': f'Régimen {estado}, sobrevendido',
                    'confianza': 0.6,
                    'tamaño': 0.05
                }
            elif rsi > 70:
                señal = {
                    'tipo': 'VENDER',
                    'razón': f'Régimen {estado}, sobrecomprado',
                    'confianza': 0.6,
                    'tamaño': 0.05
                }
        
        return señal

# Uso
estrategia = EstrategiaHMMRegimen()
estrategia.entrenar(df_historico_2años)

# En vivo
info_estado = estrategia.predecir_estado_actual(df_reciente_50días)
print(f"Estado actual: {info_estado['estado']}")
print(f"Confianza: {info_estado['confianza']:.1%}")
print(f"Prob. de cambio: {info_estado['prob_cambio']:.1%}")

señal = estrategia.generar_señal(df_actual, {'rsi': 35})
if señal:
    print(f"\nSeñal: {señal['tipo']}")
    print(f"Razón: {señal['razón']}")
```

---

## Teoría de Optimización

### 4.1 Optimización Convexa

**Intuición**: Encontrar el mejor punto donde "mejor" es no ambiguo - hay un único mínimo global.

**Visualización**:
```
Función No Convexa:          Función Convexa:
    ^                            ^
    |  /\    /\                 |      /\
    | /  \  /  \                |     /  \
    |/    \/    \               |    /    \
    +-------------->            +-------------->
Múltiples mínimos            Mínimo único global
```

**Definición Matemática**:

Una función $f: \mathbb{R}^n \to \mathbb{R}$ es **convexa** si:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

para todo $x, y \in \mathbb{R}^n$ y $\lambda \in [0, 1]$.

**Propiedad Clave**: Cualquier mínimo local es mínimo global.

---

### 4.2 Optimización de Portafolio (Markowitz)

**Problema**: Encontrar pesos óptimos para minimizar riesgo dado retorno objetivo

$$
\min_w \quad \frac{1}{2} w^T \Sigma w
$$
$$
\text{s.t.} \quad w^T \mu \geq r_{\text{objetivo}}
$$
$$
w^T \mathbf{1} = 1
$$
$$
w \geq 0
$$

donde:
- $w$: vector de pesos del portafolio
- $\Sigma$: matriz de covarianza de retornos
- $\mu$: vector de retornos esperados
- $r_{\text{objetivo}}$: retorno mínimo deseado

**Ejemplo Numérico (2 activos)**:

```
Activo A: μ_A = 8%, σ_A = 15%
Activo B: μ_B = 12%, σ_B = 25%
Correlación: ρ = 0.3

Objetivo: r_target = 10%

Matriz de covarianza:
Σ = [σ_A²           ρ·σ_A·σ_B  ]
    [ρ·σ_A·σ_B      σ_B²       ]
    
  = [0.0225    0.01125]
    [0.01125   0.0625 ]

Lagrangiano:
L(w, λ₁, λ₂) = ½wᵀΣw + λ₁(w·μ - 0.10) + λ₂(w·1 - 1)

Condiciones KKT:
∂L/∂w = Σw + λ₁μ + λ₂1 = 0

Solución:
w = -Σ⁻¹(λ₁μ + λ₂1)

Aplicando restricciones:
w_A = 0.50 (50% en activo A)
w_B = 0.50 (50% en activo B)

Retorno del portafolio: 0.5×8% + 0.5×12% = 10% ✓
Volatilidad del portafolio:
σ_p = √(w_A²σ_A² + w_B²σ_B² + 2w_Aw_Bρσ_Aσ_B)
    = √(0.25×225 + 0.25×625 + 2×0.25×0.3×15×25)
    = √(56.25 + 156.25 + 56.25)
    = 16.5%
```

**Implementación en Python**:

```python
import cvxpy as cp

def optimizar_portafolio(mu, Sigma, r_target):
    """
    Optimización de Markowitz con cvxpy
    
    Args:
        mu: Retornos esperados (n,)
        Sigma: Matriz de covarianza (n, n)
        r_target: Retorno objetivo
    
    Returns:
        pesos: Pesos óptimos
        volatilidad: Volatilidad del portafolio
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Objetivo: minimizar varianza
    objetivo = cp.Minimize(cp.quad_form(w, Sigma))
    
    # Restricciones
    restricciones = [
        w @ mu >= r_target,  # Retorno mínimo
        cp.sum(w) == 1,      # Suma a 1
        w >= 0               # Long only
    ]
    
    # Resolver
    problema = cp.Problem(objetivo, restricciones)
    problema.solve()
    
    pesos_optimos = w.value
    volatilidad = np.sqrt(pesos_optimos @ Sigma @ pesos_optimos)
    
    return pesos_optimos, volatilidad

# Ejemplo
mu = np.array([0.08, 0.12, 0.10])  # 3 activos
Sigma = np.array([[0.04, 0.01, 0.02],
                  [0.01, 0.09, 0.03],
                  [0.02, 0.03, 0.06]])
r_target = 0.10

w_opt, vol_opt = optimizar_portafolio(mu, Sigma, r_target)
print(f"Pesos óptimos: {w_opt}")
print(f"Volatilidad: {vol_opt:.2%}")
```

---

## Resumen Parte 3

**HMM**:
- Capturan estados ocultos del mercado (regímenes)
- Algoritmo de Viterbi encuentra secuencia más probable
- Baum-Welch estima parámetros automáticamente
- Útil para adaptar estrategia según régimen

**Optimización Convexa**:
- Garantiza encontrar solución global
- Base de optimización de portafolios
- Problemas cuadráticos (Markowitz) son convexos

**Aplicaciones en Trading**:
- HMM: Detección de régimen, ajuste de estrategia
- Optimización: Asignación de capital, dimensionamiento de posición

**Próxima Parte**: Guía de Ejercicios en Español
