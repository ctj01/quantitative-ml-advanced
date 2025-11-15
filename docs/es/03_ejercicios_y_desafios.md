# Ejercicios y Desafíos - Versión Español

## Estructura de los Ejercicios

Cada ejercicio incluye:
- 🎯 **Objetivo**: Qué aprenderás
- 📝 **Problema**: Descripción detallada
- 💡 **Pistas**: Ayuda para comenzar
- ✅ **Solución**: Código y explicación completa
- 🎓 **Conceptos Clave**: Lecciones principales

---

## Nivel 1: Fundamentos

### Ejercicio 1.1: Calcular ARCH(1) Manualmente

🎯 **Objetivo**: Entender cómo ARCH modela volatilidad variable en el tiempo

📝 **Problema**:
Tienes un modelo ARCH(1): $\sigma_t^2 = 0.01 + 0.3 \times r_{t-1}^2$

Datos iniciales:
- $r_0 = 2\%$ (retorno en t=0)
- $\sigma_0 = 2\%$ (volatilidad inicial)

Shocks aleatorios: $\varepsilon_1 = -0.5, \varepsilon_2 = 1.2, \varepsilon_3 = 0.3$

**Calcula**:
1. Volatilidad y retorno para t=1, 2, 3
2. ¿En qué período fue mayor la volatilidad?
3. ¿Por qué?

💡 **Pistas**:
- Usa $r_t = \sigma_t \times \varepsilon_t$
- Primero calcula $\sigma_t$, luego $r_t$
- La volatilidad depende del cuadrado del retorno anterior

<details>
<summary>✅ Ver Solución</summary>

```python
import numpy as np

# Parámetros ARCH(1)
omega = 0.01
alpha = 0.3

# Inicialización
r_0 = 0.02  # 2%
sigma_0 = 0.02  # 2%
shocks = [-0.5, 1.2, 0.3]

# Arrays para almacenar resultados
volatilidades = [sigma_0]
retornos = [r_0]

print("=== Cálculos ARCH(1) Paso a Paso ===\n")

for t, epsilon in enumerate(shocks, start=1):
    # Paso 1: Calcular volatilidad en tiempo t
    r_prev = retornos[-1]
    sigma_t_squared = omega + alpha * (r_prev ** 2)
    sigma_t = np.sqrt(sigma_t_squared)
    
    # Paso 2: Calcular retorno en tiempo t
    r_t = sigma_t * epsilon
    
    # Guardar
    volatilidades.append(sigma_t)
    retornos.append(r_t)
    
    # Mostrar
    print(f"Período t={t}:")
    print(f"  σ²_{t} = {omega} + {alpha} × ({r_prev:.4f})²")
    print(f"       = {omega} + {alpha} × {r_prev**2:.6f}")
    print(f"       = {sigma_t_squared:.6f}")
    print(f"  σ_{t} = √{sigma_t_squared:.6f} = {sigma_t:.4f} ({sigma_t*100:.2f}%)")
    print(f"  r_{t} = σ_{t} × ε_{t} = {sigma_t:.4f} × {epsilon}")
    print(f"       = {r_t:.4f} ({r_t*100:.2f}%)")
    print()

# Análisis
print("=== Resumen ===")
for t in range(len(volatilidades)):
    if t == 0:
        print(f"t={t}: σ={volatilidades[t]*100:.2f}%, r={retornos[t]*100:.2f}%")
    else:
        print(f"t={t}: σ={volatilidades[t]*100:.2f}%, r={retornos[t]*100:.2f}%, ε={shocks[t-1]}")

max_vol_idx = np.argmax(volatilidades)
print(f"\nMayor volatilidad en t={max_vol_idx}: {volatilidades[max_vol_idx]*100:.2f}%")

# Respuestas
"""
RESULTADOS:
t=0: σ=2.00%, r=2.00%
t=1: σ=10.06%, r=-5.03% (shock negativo amplificado por alta volatilidad)
t=2: σ=10.37%, r=12.44% (volatilidad sigue alta debido al gran retorno en t=1)
t=3: σ=12.10%, r=3.63% (volatilidad máxima debido al gran retorno en t=2)

MAYOR VOLATILIDAD: t=3 (12.10%)

¿POR QUÉ?
- En t=1, hubo shock negativo grande (-5.03%)
- En t=2, este shock elevó σ, que amplificó el shock positivo (+12.44%)
- En t=3, el gran retorno de t=2 causó la máxima volatilidad
- Esto muestra "clustering de volatilidad" - alta volatilidad persiste
"""
```

</details>

🎓 **Conceptos Clave**:
- ARCH captura "clustering de volatilidad"
- Los shocks grandes aumentan la volatilidad futura
- La volatilidad alta amplifica futuros shocks

---

### Ejercicio 1.2: GARCH(1,1) vs ARCH(1)

🎯 **Objetivo**: Entender por qué GARCH es más eficiente que ARCH

📝 **Problema**:
Compara dos modelos con los mismos datos:

**ARCH(1)**: $\sigma_t^2 = 0.01 + 0.3 r_{t-1}^2$

**GARCH(1,1)**: $\sigma_t^2 = 0.00001 + 0.08 r_{t-1}^2 + 0.90 \sigma_{t-1}^2$

Datos: $r_0 = 3\%$, $\sigma_0^2 = 0.0004$

Shock: $\varepsilon_1 = 1.5$, luego $\varepsilon_2 = \varepsilon_3 = 0.1$

**Calcula**:
1. Volatilidad para t=1,2,3 en ambos modelos
2. ¿Cuál modelo tiene volatilidad más persistente?
3. Calcula la varianza incondicional del GARCH

<details>
<summary>✅ Ver Solución</summary>

```python
import numpy as np

# Parámetros
omega_arch = 0.01
alpha_arch = 0.3

omega_garch = 0.00001
alpha_garch = 0.08
beta_garch = 0.90

# Inicialización
r_0 = 0.03
sigma_0_sq = 0.0004
shocks = [1.5, 0.1, 0.1]

print("=== COMPARACIÓN ARCH(1) vs GARCH(1,1) ===\n")

# ARCH(1)
print("ARCH(1) Model:")
sigma_arch = [np.sqrt(sigma_0_sq)]
r_arch = [r_0]

for t, eps in enumerate(shocks, 1):
    r_prev = r_arch[-1]
    sigma_sq = omega_arch + alpha_arch * r_prev**2
    sigma = np.sqrt(sigma_sq)
    r = sigma * eps
    
    sigma_arch.append(sigma)
    r_arch.append(r)
    
    print(f"  t={t}: σ²={sigma_sq:.6f}, σ={sigma*100:.2f}%, r={r*100:.2f}%")

# GARCH(1,1)
print("\nGARCH(1,1) Model:")
sigma_garch = [np.sqrt(sigma_0_sq)]
r_garch = [r_0]

for t, eps in enumerate(shocks, 1):
    r_prev = r_garch[-1]
    sigma_prev_sq = sigma_garch[-1]**2
    sigma_sq = omega_garch + alpha_garch * r_prev**2 + beta_garch * sigma_prev_sq
    sigma = np.sqrt(sigma_sq)
    r = sigma * eps
    
    sigma_garch.append(sigma)
    r_garch.append(r)
    
    print(f"  t={t}: σ²={sigma_sq:.6f}, σ={sigma*100:.2f}%, r={r*100:.2f}%")

# Análisis de persistencia
print("\n=== ANÁLISIS DE PERSISTENCIA ===")
print(f"ARCH: σ_1={sigma_arch[1]*100:.2f}% → σ_3={sigma_arch[3]*100:.2f}%")
print(f"      Decaimiento: {(sigma_arch[3]/sigma_arch[1] - 1)*100:.1f}%")
print(f"\nGARCH: σ_1={sigma_garch[1]*100:.2f}% → σ_3={sigma_garch[3]*100:.2f}%")
print(f"       Decaimiento: {(sigma_garch[3]/sigma_garch[1] - 1)*100:.1f}%")

# Varianza incondicional GARCH
var_incond = omega_garch / (1 - alpha_garch - beta_garch)
sigma_incond = np.sqrt(var_incond)
print(f"\n=== VARIANZA INCONDICIONAL GARCH ===")
print(f"Fórmula: ω / (1 - α - β)")
print(f"       = {omega_garch} / (1 - {alpha_garch} - {beta_garch})")
print(f"       = {omega_garch} / {1 - alpha_garch - beta_garch}")
print(f"       = {var_incond:.6f}")
print(f"σ̄ = {sigma_incond*100:.2f}%")

print("\n=== CONCLUSIONES ===")
print("1. GARCH tiene volatilidad más PERSISTENTE")
print("   - Después del shock en t=1, GARCH decae más lentamente")
print("   - ARCH: revierte rápido a la media")
print("   - GARCH: mantiene 'memoria' de volatilidad pasada vía β")
print("\n2. GARCH es más PARSIMONIOSO")
print("   - ARCH necesitaría muchos lags para capturar persistencia")
print("   - GARCH(1,1) lo logra con solo 3 parámetros")
print("\n3. PERSISTENCIA = α + β")
print(f"   - GARCH: {alpha_garch + beta_garch} (muy persistente)")
print("   - Cerca de 1 = volatilidad dura mucho tiempo")
```

</details>

🎓 **Conceptos Clave**:
- GARCH añade "momentum" de volatilidad (término β)
- Mayor persistencia = decaimiento más lento
- α + β cerca de 1 → volatilidad muy persistente

---

## Nivel 2: Implementación

### Ejercicio 2.1: Estimar GARCH con MLE

🎯 **Objetivo**: Implementar estimación de máxima verosimilitud para GARCH

📝 **Problema**:
Descarga datos de S&P 500 (últimos 2 años) y:
1. Calcula retornos diarios
2. Implementa la función de log-verosimilitud para GARCH(1,1)
3. Usa `scipy.optimize.minimize` para encontrar parámetros óptimos
4. Compara con librería `arch`

💡 **Pistas**:
- Log-verosimilitud: $\sum_{t=1}^{T} \left[-\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma_t^2) - \frac{r_t^2}{2\sigma_t^2}\right]$
- Necesitas iterar para calcular cada $\sigma_t^2$
- Restricciones: ω > 0, α ≥ 0, β ≥ 0, α + β < 1

<details>
<summary>✅ Ver Solución</summary>

```python
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from arch import arch_model

# 1. Descargar datos
print("Descargando datos de S&P 500...")
data = yf.download('SPY', start='2022-01-01', end='2024-01-01')
returns = data['Close'].pct_change().dropna() * 100  # En %

print(f"Retornos: {len(returns)} días")
print(f"Media: {returns.mean():.3f}%")
print(f"Std: {returns.std():.3f}%")

# 2. Función de log-verosimilitud negativa
def garch_likelihood_negativa(params, returns):
    """
    Log-verosimilitud negativa para GARCH(1,1)
    (negativa porque scipy minimiza)
    """
    omega, alpha, beta = params
    
    # Validar restricciones
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10  # Penalización grande
    
    T = len(returns)
    sigma2 = np.zeros(T)
    
    # Inicializar con varianza incondicional
    sigma2[0] = np.var(returns)
    
    # Calcular volatilidades
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Log-verosimilitud
    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2
    )
    
    return -log_likelihood  # Negativa para minimización

# 3. Estimar con scipy
print("\n=== ESTIMACIÓN MANUAL (scipy) ===")

# Valores iniciales (típicos)
params_init = [0.01, 0.05, 0.90]

# Restricciones
bounds = [(1e-8, None), (0, 1), (0, 1)]

# Optimizar
result = minimize(
    garch_likelihood_negativa,
    params_init,
    args=(returns.values,),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000}
)

omega_est, alpha_est, beta_est = result.x

print(f"Parámetros estimados:")
print(f"  ω = {omega_est:.6f}")
print(f"  α = {alpha_est:.6f}")
print(f"  β = {beta_est:.6f}")
print(f"  α + β = {alpha_est + beta_est:.6f}")
print(f"Log-verosimilitud: {-result.fun:.2f}")

# Calcular varianza incondicional
var_incond = omega_est / (1 - alpha_est - beta_est)
print(f"Varianza incondicional: {var_incond:.6f}")
print(f"Volatilidad anual: {np.sqrt(var_incond * 252):.2f}%")

# 4. Comparar con arch library
print("\n=== COMPARACIÓN CON LIBRERÍA ARCH ===")

model_arch = arch_model(returns, vol='Garch', p=1, q=1)
result_arch = model_arch.fit(disp='off')

print(result_arch.summary())

print("\n=== DIFERENCIA ===")
params_arch = result_arch.params
print(f"ω: Manual={omega_est:.6f}, Arch={params_arch['omega']:.6f}")
print(f"α: Manual={alpha_est:.6f}, Arch={params_arch['alpha[1]']:.6f}")
print(f"β: Manual={beta_est:.6f}, Arch={params_arch['beta[1]']:.6f}")

# 5. Visualizar volatilidad estimada
import matplotlib.pyplot as plt

# Calcular volatilidad con parámetros estimados
T = len(returns)
sigma2_manual = np.zeros(T)
sigma2_manual[0] = np.var(returns)

for t in range(1, T):
    sigma2_manual[t] = omega_est + alpha_est * returns.values[t-1]**2 + beta_est * sigma2_manual[t-1]

sigma_manual = np.sqrt(sigma2_manual)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(returns.index, sigma_manual, label='Volatilidad GARCH', linewidth=2)
plt.plot(returns.index, np.abs(returns.values), alpha=0.3, label='|Retornos|')
plt.title('Volatilidad Estimada GARCH(1,1) - S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad (%)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('volatilidad_garch_sp500.png', dpi=150)
print("\nGráfico guardado: volatilidad_garch_sp500.png")
```

</details>

🎓 **Conceptos Clave**:
- MLE encuentra parámetros que maximizan probabilidad de observar los datos
- Optimización numérica necesaria (no hay solución cerrada)
- Restricciones importantes para estacionariedad

---

### Ejercicio 2.2: Comparar GARCH vs EGARCH

🎯 **Objetivo**: Ver cómo EGARCH captura efectos de apalancamiento

📝 **Problema**:
1. Estima GARCH(1,1) y EGARCH(1,1) en S&P 500
2. Simula un shock negativo de -5% y uno positivo de +5%
3. Compara cómo afectan la volatilidad en cada modelo
4. ¿Cuál modelo muestra asimetría?

<details>
<summary>✅ Ver Solución</summary>

```python
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

# Descargar datos
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')
returns = data['Close'].pct_change().dropna() * 100

print("=== ESTIMACIÓN DE MODELOS ===\n")

# Estimar GARCH(1,1)
model_garch = arch_model(returns, vol='Garch', p=1, q=1)
result_garch = model_garch.fit(disp='off')
print("GARCH(1,1):")
print(result_garch.summary().tables[1])

# Estimar EGARCH(1,1)
model_egarch = arch_model(returns, vol='EGARCH', p=1, q=1)
result_egarch = model_egarch.fit(disp='off')
print("\nEGARCH(1,1):")
print(result_egarch.summary().tables[1])

# Extraer parámetros
omega_g = result_garch.params['omega']
alpha_g = result_garch.params['alpha[1]']
beta_g = result_garch.params['beta[1]']

omega_e = result_egarch.params['omega']
alpha_e = result_egarch.params['alpha[1]']
gamma_e = result_egarch.params['gamma[1]']
beta_e = result_egarch.params['beta[1]']

print(f"\n=== PARÁMETROS CLAVE ===")
print(f"GARCH:  α={alpha_g:.4f}, β={beta_g:.4f}, α+β={alpha_g+beta_g:.4f}")
print(f"EGARCH: α={alpha_e:.4f}, γ={gamma_e:.4f}, β={beta_e:.4f}")
print(f"\n¿γ < 0? {gamma_e < 0} → {'SÍ hay' if gamma_e < 0 else 'NO hay'} efecto de apalancamiento")

# Simulación de shocks
print("\n=== SIMULACIÓN DE SHOCKS ===\n")

def simular_shock_garch(shock, sigma_inicial, omega, alpha, beta, periodos=10):
    """Simular impacto de shock en GARCH"""
    sigma2 = np.zeros(periodos + 1)
    sigma2[0] = sigma_inicial**2
    returns_sim = np.zeros(periodos + 1)
    returns_sim[0] = shock
    
    for t in range(1, periodos + 1):
        sigma2[t] = omega + alpha * returns_sim[t-1]**2 + beta * sigma2[t-1]
        returns_sim[t] = 0  # Sin más shocks
    
    return np.sqrt(sigma2)

def simular_shock_egarch(shock, sigma_inicial, omega, alpha, gamma, beta, periodos=10):
    """Simular impacto de shock en EGARCH"""
    log_sigma2 = np.zeros(periodos + 1)
    log_sigma2[0] = np.log(sigma_inicial**2)
    
    for t in range(1, periodos + 1):
        if t == 1:
            z = shock / sigma_inicial
        else:
            z = 0
        
        log_sigma2[t] = omega + alpha * abs(z) + gamma * z + beta * log_sigma2[t-1]
    
    return np.exp(log_sigma2 / 2)

# Volatilidad inicial (promedio)
sigma_init = returns.std()

# Shock negativo (-5%)
shock_neg = -5.0
sigma_garch_neg = simular_shock_garch(shock_neg, sigma_init, omega_g, alpha_g, beta_g)
sigma_egarch_neg = simular_shock_egarch(shock_neg, sigma_init, omega_e, alpha_e, gamma_e, beta_e)

# Shock positivo (+5%)
shock_pos = 5.0
sigma_garch_pos = simular_shock_garch(shock_pos, sigma_init, omega_g, alpha_g, beta_g)
sigma_egarch_pos = simular_shock_egarch(shock_pos, sigma_init, omega_e, alpha_e, gamma_e, beta_e)

# Comparar
print(f"Volatilidad inicial: {sigma_init:.2f}%\n")

print("Después de shock NEGATIVO de -5%:")
print(f"  GARCH día 1:  {sigma_garch_neg[1]:.2f}% (+{(sigma_garch_neg[1]/sigma_init-1)*100:.1f}%)")
print(f"  EGARCH día 1: {sigma_egarch_neg[1]:.2f}% (+{(sigma_egarch_neg[1]/sigma_init-1)*100:.1f}%)")

print("\nDespués de shock POSITIVO de +5%:")
print(f"  GARCH día 1:  {sigma_garch_pos[1]:.2f}% (+{(sigma_garch_pos[1]/sigma_init-1)*100:.1f}%)")
print(f"  EGARCH día 1: {sigma_egarch_pos[1]:.2f}% (+{(sigma_egarch_pos[1]/sigma_init-1)*100:.1f}%)")

diff_garch = sigma_garch_neg[1] - sigma_garch_pos[1]
diff_egarch = sigma_egarch_neg[1] - sigma_egarch_pos[1]

print(f"\n=== ASIMETRÍA ===")
print(f"GARCH:  Diferencia = {diff_garch:.3f}% (casi simétrico)")
print(f"EGARCH: Diferencia = {diff_egarch:.3f}% (asimétrico)")
print(f"\n→ Shock negativo aumenta volatilidad {diff_egarch:.2f}% MÁS en EGARCH")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GARCH
axes[0].plot(sigma_garch_neg, 'r-', label='Shock -5%', linewidth=2)
axes[0].plot(sigma_garch_pos, 'g-', label='Shock +5%', linewidth=2)
axes[0].axhline(sigma_init, color='k', linestyle='--', alpha=0.5, label='Vol inicial')
axes[0].set_title('GARCH(1,1) - Respuesta Simétrica')
axes[0].set_xlabel('Días después del shock')
axes[0].set_ylabel('Volatilidad (%)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# EGARCH
axes[1].plot(sigma_egarch_neg, 'r-', label='Shock -5%', linewidth=2)
axes[1].plot(sigma_egarch_pos, 'g-', label='Shock +5%', linewidth=2)
axes[1].axhline(sigma_init, color='k', linestyle='--', alpha=0.5, label='Vol inicial')
axes[1].set_title('EGARCH(1,1) - Respuesta Asimétrica')
axes[1].set_xlabel('Días después del shock')
axes[1].set_ylabel('Volatilidad (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('garch_vs_egarch_asimetria.png', dpi=150)
print("\nGráfico guardado: garch_vs_egarch_asimetria.png")
```

</details>

🎓 **Conceptos Clave**:
- EGARCH captura "efecto de apalancamiento"
- Shocks negativos aumentan más la volatilidad que shocks positivos
- Parámetro γ < 0 indica asimetría
- Crítico para mercados de acciones (crashes vs rallies)

---

## Nivel 3: Trading Avanzado

### Ejercicio 3.1: Sistema de Trading con GARCH + RSI

🎯 **Objetivo**: Construir estrategia combinando volatilidad y momentum

📝 **Problema**:
Crea un sistema que:
1. Use GARCH para pronosticar volatilidad
2. Calcule RSI para momentum
3. Reglas:
   - COMPRAR: RSI < 30 Y volatilidad futura < percentil 50
   - VENDER: RSI > 70 Y volatilidad futura > percentil 75
4. Dimensionamiento: inversamente proporcional a volatilidad
5. Backtest en S&P 500 (2020-2024)

💡 **Pistas**:
- Usa ventana rodante para estimar GARCH
- Recalcula percentiles de volatilidad cada mes
- Comisiones: 0.1% por operación

<details>
<summary>✅ Ver Solución Completa</summary>

```python
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

class SistemaGARCH_RSI:
    def __init__(self, capital_inicial=100000, comision=0.001):
        self.capital_inicial = capital_inicial
        self.capital = capital_inicial
        self.comision = comision
        self.posicion = 0
        self.historial = []
        
    def calcular_rsi(self, precios, periodo=14):
        """Calcular RSI"""
        delta = precios.diff()
        ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def pronosticar_volatilidad(self, retornos, horizon=5):
        """Estimar GARCH y pronosticar"""
        try:
            model = arch_model(retornos, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            forecast = result.forecast(horizon=horizon)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :])
            return vol_forecast[0]  # Pronóstico 1-día adelante
        except:
            return retornos.std()
    
    def ejecutar_backtest(self, df):
        """Ejecutar backtest completo"""
        print("=== INICIANDO BACKTEST ===\n")
        
        # Preparar datos
        df = df.copy()
        df['returns'] = df['Close'].pct_change() * 100
        df['rsi'] = self.calcular_rsi(df['Close'])
        
        # Ventana para estimar GARCH
        ventana_garch = 252  # 1 año
        
        # Iterar sobre datos
        for i in range(ventana_garch, len(df)):
            fecha = df.index[i]
            precio = df['Close'].iloc[i]
            rsi = df['rsi'].iloc[i]
            
            # Estimar volatilidad con ventana rodante
            retornos_ventana = df['returns'].iloc[i-ventana_garch:i]
            vol_forecast = self.pronosticar_volatilidad(retornos_ventana)
            
            # Calcular percentiles de volatilidad (últimos 60 días)
            volatilidades_recientes = [
                self.pronosticar_volatilidad(df['returns'].iloc[j-ventana_garch:j])
                for j in range(max(ventana_garch, i-60), i)
            ]
            vol_p50 = np.percentile(volatilidades_recientes, 50)
            vol_p75 = np.percentile(volatilidades_recientes, 75)
            
            # Lógica de trading
            señal = None
            
            if self.posicion == 0:  # Sin posición
                if rsi < 30 and vol_forecast < vol_p50:
                    # Señal de COMPRA
                    tamaño = self.capital / precio
                    tamaño *= (vol_p50 / vol_forecast)  # Ajuste por volatilidad
                    tamaño = min(tamaño, self.capital / precio)  # No sobre-apalancar
                    
                    costo = tamaño * precio * (1 + self.comision)
                    if costo <= self.capital:
                        self.posicion = tamaño
                        self.capital -= costo
                        señal = 'COMPRA'
                        
            elif self.posicion > 0:  # Con posición larga
                if rsi > 70 and vol_forecast > vol_p75:
                    # Señal de VENTA
                    ingresos = self.posicion * precio * (1 - self.comision)
                    self.capital += ingresos
                    self.posicion = 0
                    señal = 'VENTA'
            
            # Guardar estado
            valor_portafolio = self.capital + self.posicion * precio
            self.historial.append({
                'fecha': fecha,
                'precio': precio,
                'rsi': rsi,
                'vol_forecast': vol_forecast,
                'vol_p50': vol_p50,
                'vol_p75': vol_p75,
                'posicion': self.posicion,
                'capital': self.capital,
                'valor_portafolio': valor_portafolio,
                'señal': señal
            })
            
            # Mostrar señales
            if señal and (i - ventana_garch) % 30 == 0:  # Cada ~30 días con señal
                print(f"{fecha.date()}: {señal} - RSI={rsi:.1f}, Vol={vol_forecast:.2f}%")
        
        return pd.DataFrame(self.historial)
    
    def analizar_resultados(self, df_resultados):
        """Analizar performance del backtest"""
        print("\n=== RESULTADOS DEL BACKTEST ===\n")
        
        valor_final = df_resultados['valor_portafolio'].iloc[-1]
        retorno_total = (valor_final / self.capital_inicial - 1) * 100
        
        # Calcular métricas
        retornos_diarios = df_resultados['valor_portafolio'].pct_change().dropna()
        sharpe = retornos_diarios.mean() / retornos_diarios.std() * np.sqrt(252)
        
        # Drawdown
        rolling_max = df_resultados['valor_portafolio'].expanding().max()
        drawdown = (df_resultados['valor_portafolio'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Contar operaciones
        señales = df_resultados[df_resultados['señal'].notna()]
        num_operaciones = len(señales)
        compras = len(señales[señales['señal'] == 'COMPRA'])
        ventas = len(señales[señales['señal'] == 'VENTA'])
        
        print(f"Capital Inicial:  ${self.capital_inicial:,.2f}")
        print(f"Valor Final:      ${valor_final:,.2f}")
        print(f"Retorno Total:    {retorno_total:.2f}%")
        print(f"Sharpe Ratio:     {sharpe:.2f}")
        print(f"Max Drawdown:     {max_drawdown:.2f}%")
        print(f"\nOperaciones:      {num_operaciones}")
        print(f"  Compras:        {compras}")
        print(f"  Ventas:         {ventas}")
        
        # Comparar con Buy & Hold
        precio_inicial = df_resultados['precio'].iloc[0]
        precio_final = df_resultados['precio'].iloc[-1]
        retorno_bh = (precio_final / precio_inicial - 1) * 100
        
        print(f"\nBuy & Hold:       {retorno_bh:.2f}%")
        print(f"Exceso:           {retorno_total - retorno_bh:.2f}%")
        
        return df_resultados

# Ejecutar backtest
print("Descargando datos...")
data = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)

sistema = SistemaGARCH_RSI(capital_inicial=100000)
resultados = sistema.ejecutar_backtest(data)
resultados = sistema.analizar_resultados(resultados)

# Visualización
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Precio y señales
axes[0].plot(resultados['fecha'], resultados['precio'], label='S&P 500', linewidth=1.5)
compras = resultados[resultados['señal'] == 'COMPRA']
ventas = resultados[resultados['señal'] == 'VENTA']
axes[0].scatter(compras['fecha'], compras['precio'], color='green', marker='^', s=100, label='Compra', zorder=5)
axes[0].scatter(ventas['fecha'], ventas['precio'], color='red', marker='v', s=100, label='Venta', zorder=5)
axes[0].set_title('Señales de Trading')
axes[0].set_ylabel('Precio')
axes[0].legend()
axes[0].grid(alpha=0.3)

# RSI
axes[1].plot(resultados['fecha'], resultados['rsi'], label='RSI', color='purple')
axes[1].axhline(70, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(30, color='g', linestyle='--', alpha=0.5)
axes[1].set_title('RSI')
axes[1].set_ylabel('RSI')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Valor del portafolio
axes[2].plot(resultados['fecha'], resultados['valor_portafolio'], label='Estrategia', linewidth=2)
bh_valores = 100000 * (resultados['precio'] / resultados['precio'].iloc[0])
axes[2].plot(resultados['fecha'], bh_valores, label='Buy & Hold', linestyle='--', alpha=0.7)
axes[2].set_title('Valor del Portafolio')
axes[2].set_xlabel('Fecha')
axes[2].set_ylabel('Valor ($)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_garch_rsi.png', dpi=150)
print("\nGráfico guardado: backtest_garch_rsi.png")
```

</details>

🎓 **Conceptos Clave**:
- Combinación de volatilidad y momentum mejora señales
- Dimensionamiento basado en volatilidad reduce riesgo
- Ventana rodante evita sesgo de anticipación
- Comparar siempre con benchmark (Buy & Hold)

---

## Resumen

Has completado ejercicios que cubren:
- ✅ Cálculos manuales de ARCH/GARCH
- ✅ Estimación MLE con scipy
- ✅ Comparación GARCH vs EGARCH
- ✅ Sistema de trading completo
- ✅ Backtesting riguroso

**Próximos Pasos**:
1. Implementar más modelos (HMM, Transformers)
2. Añadir más indicadores técnicos
3. Optimizar hiperparámetros
4. Análisis de riesgo avanzado

**Recursos Adicionales**:
- Notebook 01: Implementaciones detalladas
- Notebook 08: Proyecto completo integrado
- docs/: Teoría matemática completa
