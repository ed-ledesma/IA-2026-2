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
