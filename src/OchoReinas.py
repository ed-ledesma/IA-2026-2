"""
=============================================================================
UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO (UNAM)
Facultad de Ciencias 
Materia: Inteligencia Artificial
Docente: Dra. Jessica Sarahi Méndez Rincón
Ayudante de Laboratorio: Diego Eduardo Peña Villegas
Alumno: [Nombre del Alumno]
Año escolar: 2026-2
Copyright: (c) 2025 UNAM - MIT License
Version: 1.0
This software is for educational purposes.  
-----------------------------------------------------------------------------
UNAM IA Library: A professional toolkit for AI developed at UNAM.
=============================================================================
"""

import matplotlib.pyplot as plt

def es_seguro(tablero, fila, columna):
    # Verificar si es seguro colocar una reina en la fila, columna
    for i in range(fila):
        if tablero[i] == columna or \
           tablero[i] - i == columna - fila or \
           tablero[i] + i == columna + fila:
            return False
    return True

def resolver_8_reinas(tablero, fila):
    if fila == 8:
        return True
    for columna in range(8):
        if es_seguro(tablero, fila, columna):
            tablero[fila] = columna
            if resolver_8_reinas(tablero, fila + 1):
                return True
            tablero[fila] = -1
    return False

def dibujar_tablero(tablero):
    plt.figure(figsize=(8, 8))
    plt.matshow([[tablero[i] == j for j in range(8)] for i in range(8)], cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Inicializar el tablero
tablero = [-1] * 8

# Resolver el problema de las 8 reinas
if resolver_8_reinas(tablero, 0):
    print("Solución encontrada:")
    for fila, columna in enumerate(tablero):
        print(f"Reina en fila {fila} y columna {columna}")
    dibujar_tablero(tablero)
else:
    print("No se encontró solución.")