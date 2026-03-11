# Algoritmos para Inteligencia Artificial

## Ocho Reinas

El problema de las Ocho Reinas es un problema clásico en las Ciencias de la Computación. Consiste
en colocar ocho reinas en un tablero de ajedrez estándar de 8x8 de tal manera que ningún para de
reinas se amenace entre sı́, más especı́ficamente, una solución requiere que ningún par de reinas
comparta la misma fila, columna o diagonal.

*Pseudocódigo*

La forma clásica de resolver el problema es a través del uso de backtracking. Presentamos aquí una implementación recursiva.

OchoReinas

```text
Algoritmo NReinas(n)
    tablero ← arreglo de tamaño n
    resolver(0)

función resolver(fila)
    si fila = n entonces
        imprimir tablero
        regresar

    para columna ← 0 hasta n-1
        si esSeguro(tablero, fila, columna) entonces
            tablero[fila] ← columna
            resolver(fila + 1)
```
