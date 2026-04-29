"""
=============================================================================
UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO (UNAM)
Facultad de Ciencias 
Materia: Inteligencia Artificial
Docente: Dra. Jessica Sarahi Méndez Rincón
Ayudante de Laboratorio: Diego Eduardo Peña Villegas
Alumno: Eduardo Ledesma Cuevas
Año escolar: 2026-2
Copyright: (c) 2025 UNAM - MIT License
Version: 1.0
This software is for educational purposes.  
-----------------------------------------------------------------------------
UNAM IA Library: A professional toolkit for AI developed at UNAM.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class RegresionLineal:
    """
    Regresión Lineal entrenada con Descenso de Gradiente.

    Parámetros
    ----------
    alpha      : float -- Tasa de aprendizaje.
    iteraciones: int   -- Número de épocas de entrenamiento.
    """

    def __init__(self, alpha: float = 0.01, iteraciones: int = 1000):
        self.alpha       = alpha
        self.iteraciones = iteraciones
        self.w           = None
        self.b           = None
        self.historial_perdida = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrena el modelo ajustando w y b mediante descenso de gradiente."""
        n_muestras, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.iteraciones):
            y_pred = np.dot(X, self.w) + self.b

            # Gradientes del MSE
            dw = (1 / n_muestras) * np.dot(X.T, (y_pred - y))
            db = (1 / n_muestras) * np.sum(y_pred - y)

            # Actualización de parámetros
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            # Registrar pérdida para análisis de convergencia
            perdida = np.mean((y_pred - y) ** 2)
            self.historial_perdida.append(perdida)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones: y_hat = X·w + b"""
        return np.dot(X, self.w) + self.b


if __name__ == "__main__":
    # Datos sintéticos
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    modelo = RegresionLineal(alpha=0.01, iteraciones=1000)
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)
    print(f"MSE  : {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")

    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.scatter(X_test, y_test, color="steelblue", alpha=0.6, label="Datos reales")
    ax1.plot(X_test, y_pred, color="tomato", linewidth=2, label="Predicción")
    ax1.set_title("Regresión Lineal — Ajuste", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(modelo.historial_perdida, color="purple")
    ax2.set_title("Convergencia del MSE", fontweight="bold")
    ax2.set_xlabel("Iteración"); ax2.set_ylabel("MSE")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()
