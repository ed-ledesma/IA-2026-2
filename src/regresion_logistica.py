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

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report


class RegresionLogistica:
    """
    Clasificador binario basado en la función sigmoide,
    optimizado con Descenso de Gradiente sobre la Entropía Cruzada.

    Parámetros
    ----------
    alpha      : float -- Tasa de aprendizaje.
    iteraciones: int   -- Número de épocas.
    """

    def __init__(self, alpha: float = 0.01, iteraciones: int = 1000):
        self.alpha       = alpha
        self.iteraciones = iteraciones
        self.w           = None
        self.b           = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """σ(z) = 1 / (1 + e^{-z}) — acota la salida en (0, 1)."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrena el clasificador mediante descenso de gradiente."""
        n_muestras, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.iteraciones):
            modelo_lineal = np.dot(X, self.w) + self.b
            y_pred        = self._sigmoid(modelo_lineal)

            # Gradientes de la Entropía Cruzada
            dw = (1 / n_muestras) * np.dot(X.T, (y_pred - y))
            db = (1 / n_muestras) * np.sum(y_pred - y)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna la probabilidad P(y=1|x) para cada muestra."""
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Clasifica usando umbral de decisión 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                n_informative=2, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RegresionLogistica(alpha=0.1, iteraciones=1000)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(f"Exactitud: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Clase 0","Clase 1"]))

    # Curva sigmoide
    z_vals = np.linspace(-10, 10, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(z_vals, 1/(1+np.exp(-z_vals)), color="steelblue", linewidth=2)
    plt.axhline(0.5, color="tomato", linestyle="--", label="Umbral 0.5")
    plt.title("Función Sigmoide", fontweight="bold")
    plt.xlabel("z"); plt.ylabel("σ(z)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()