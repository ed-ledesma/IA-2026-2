# Algoritmos para Inteligencia Artificial

## Ocho Reinas

El problema de las Ocho Reinas es un problema clásico en las Ciencias de la Computación. Consiste
en colocar ocho reinas en un tablero de ajedrez estándar de 8x8 de tal manera que ningún para de
reinas se amenace entre sı́, más especı́ficamente, una solución requiere que ningún par de reinas
comparta la misma fila, columna o diagonal.

*Pseudocódigo*

La forma clásica de resolver el problema es a través del uso de backtracking. Presentamos aquí una implementación recursiva.

es_seguro(tablero, fila, columna)
```text
    para i ← 0 hasta fila-1

        si tablero[i] = columna
            retornar FALSO

        si tablero[i] - i = columna - fila
            retornar FALSO

        si tablero[i] + i = columna + fila
            retornar FALSO

    retornar VERDADERO
```

 resolver_8_reinas(tablero, fila)

```text
    si fila = N
        retornar VERDADERO

    para columna ← 0 hasta N-1

        si ES_SEGURO(tablero, fila, columna)

            tablero[fila] ← columna

            si RESOLVER_REINAS(tablero, fila + 1)
                retornar VERDADERO

            tablero[fila] ← -1

    retornar FALSO
```

## Regresión Lineal

### Fundamentos Matemáticos

El modelo predice una salida continua $\hat{y}$ como combinación lineal de las características:

$$\hat{y} = X\mathbf{w} + b$$

La **función de pérdida** es el Error Cuadrático Medio (MSE):

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

Los **gradientes** para descenso de gradiente son:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{2}{n} X^T (\hat{y} - y) \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n}\sum(\hat{y} - y)$$

**Actualización de parámetros:**

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{w}} \qquad b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

donde $\alpha$ es la **tasa de aprendizaje**.

**Complejidad:** Entrenamiento $O(n \cdot d \cdot T)$, Predicción $O(d)$, donde $T$ = iteraciones.

### Descripción Formal

La Regresión Lineal ajusta un hiperplano en el espacio de características que minimiza la suma de los errores cuadráticos entre las predicciones y los valores reales. Se optimiza mediante **Descenso de Gradiente** (Gradient Descent): se calcula el gradiente del error con respecto a cada parámetro y se actualizan en la dirección opuesta al gradiente.

### Pseudocódigo

```
ALGORITMO: Regresión Lineal con Descenso de Gradiente
=====================================================
INICIALIZAR w ← 0, b ← 0

PARA t en [1 ... T_iteraciones]:
    y_pred ← X · w + b
    dw     ← (1/n) · Xᵀ · (y_pred − y)
    db     ← (1/n) · SUMA(y_pred − y)
    w      ← w − α · dw
    b      ← b − α · db
FIN PARA

FUNCIÓN predecir(X):
    RETORNAR X · w + b
```
"""


## Regresión Logística

### Fundamentos Matemáticos

Transforma la salida lineal en una **probabilidad** usando la función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = X\mathbf{w} + b$$

La **función de pérdida** es la Entropía Cruzada Binaria (Log-Loss):

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Los gradientes son idénticos en forma a la Regresión Lineal:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{n} X^T(\hat{y} - y) \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n}\sum(\hat{y} - y)$$

**Umbral de decisión:** $\hat{y} \geq 0.5 \Rightarrow$ Clase 1, de lo contrario Clase 0.

**Complejidad:** Entrenamiento $O(n \cdot d \cdot T)$, Predicción $O(d)$.

### Descripción Formal

La Regresión Logística es un clasificador binario que modela la probabilidad posterior $P(y=1 \mid x)$. A diferencia de la Regresión Lineal, usa la función sigmoide para acotar la salida en $[0,1]$. Se optimiza minimizando la Entropía Cruzada mediante Descenso de Gradiente.

### Pseudocódigo

```
ALGORITMO: Regresión Logística
==============================
INICIALIZAR w ← 0, b ← 0

PARA t en [1 ... T_iteraciones]:
    z      ← X · w + b
    y_pred ← σ(z) = 1 / (1 + exp(−z))
    dw     ← (1/n) · Xᵀ · (y_pred − y)
    db     ← (1/n) · SUMA(y_pred − y)
    w      ← w − α · dw
    b      ← b − α · db
FIN PARA

FUNCIÓN predecir(X):
    probabilidades ← σ(X·w + b)
    RETORNAR [1 si p >= 0.5 else 0 para p en probabilidades]
```

## Naive Bayes

### Fundamentos Matemáticos

Aplica el **Teorema de Bayes** para clasificación, asumiendo independencia condicional entre características:

$$P(y \mid \mathbf{x}) = \frac{P(y) \cdot P(\mathbf{x} \mid y)}{P(\mathbf{x})}$$

Bajo el supuesto de independencia *naive*, la verosimilitud se factoriza:

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{d} P(x_j \mid y)$$

La **regla de decisión** selecciona la clase más probable:

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} \left[ \log P(y) + \sum_{j=1}^{d} \log P(x_j \mid y) \right]$$

Se usa la suma de logaritmos para evitar *underflow* numérico. Para características continuas, $P(x_j \mid y)$ se modela con una distribución Gaussiana:

$$P(x_j \mid y) = \frac{1}{\sqrt{2\pi\sigma_{j,y}^2}} \exp\!\left(-\frac{(x_j - \mu_{j,y})^2}{2\sigma_{j,y}^2}\right)$$

donde $\mu_{j,y}$ y $\sigma_{j,y}^2$ son la media y varianza de la característica $j$ en la clase $y$.

Para características discretas se usa **suavizado de Laplace** ($\alpha = 1$) para evitar probabilidades cero:

$$P(x_j = v \mid y) = \frac{\text{count}(x_j = v,\, y) + \alpha}{\text{count}(y) + \alpha \cdot V_j}$$

donde $V_j$ es el número de valores posibles de la característica $j$.

**Complejidad:** Entrenamiento $O(n \cdot d)$, Predicción $O(d \cdot |\mathcal{Y}|)$.

### Descripción Formal

Naive Bayes es un clasificador probabilístico generativo basado en el Teorema de Bayes. Durante el entrenamiento estima, para cada clase, la probabilidad a priori $P(y)$ y los parámetros de la distribución condicional $P(x_j \mid y)$ de cada característica. En la predicción combina estas estimaciones para calcular la probabilidad posterior de cada clase y retorna la de mayor valor. A pesar del supuesto de independencia condicional ---raramente cumplido en la práctica--- el clasificador es sorprendentemente robusto y eficiente, especialmente en tareas de clasificación de texto.

### Pseudocódigo

```
ALGORITMO: Naive Bayes (Gaussiano)
==================================
ENTRENAMIENTO:

PARA cada clase y en Y:
    prior[y]  ← count(y) / n
    PARA cada característica j en [1 ... d]:
        μ[j][y] ← MEDIA(x_j | clase = y)
        σ²[j][y] ← VARIANZA(x_j | clase = y)

FUNCIÓN predecir(x):
    PARA cada clase y en Y:
        log_post[y] ← log(prior[y])
        PARA cada característica j en [1 ... d]:
            log_post[y] ← log_post[y] + log_gaussiana(x_j; μ[j][y], σ²[j][y])
    RETORNAR y* = ARGMAX_y log_post[y]

FUNCIÓN log_gaussiana(x; μ, σ²):
    RETORNAR −0.5 · log(2π · σ²) − (x − μ)² / (2 · σ²)
```