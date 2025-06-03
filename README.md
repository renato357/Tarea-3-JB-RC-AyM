# 🧠 Tarea 3 - Metaheurísticas: Evolución Diferencial (DE)

Este repositorio contiene el desarrollo, implementación y análisis del algoritmo de **Evolución Diferencial (DE)** para la optimización de funciones multimodales. La tarea fue realizada en el contexto del curso de (junio 2025).


## 📂 Estructura del repositorio
```
.
├── Informe/                               # Informe final de la tarea en PDF
│   └── Tarea_3_JB_RC.pdf
├── de_optimizer.py                        # Implementación del algoritmo DE
├── aco_optimizer.py                       # Implementación de ACO (fase exploratoria)
├── ba_optimizer.py                        # Implementación de BA (fase exploratoria)
├── ga_optimizer.py                        # Implementación de GA (fase exploratoria)
├── pso_optimizer.py                       # Implementación de PSO (fase exploratoria)
├── main_graficos_de.py                    # Script principal para generar gráficos DE
├── main_graficos2.py                      # Script alternativo de visualización
├── graficos_de_optimizer.py               # Generación de gráficos DE
├── graficos_aco_optimizer.py              # Generación de gráficos ACO
├── graficos_ba_optimizer.py               # Generación de gráficos BA
├── graficos_ga_optimizer.py               # Generación de gráficos GA
├── graficos_pso_optimizer.py              # Generación de gráficos PSO
├── de_output_resumen/
│   └── resumen_estadisticas_DE_semillas.txt   # Estadísticas de DE por configuración y función
├── outputs/
│   └── resumen_mejores_configuraciones.txt    # Comparativa de mejores configuraciones
├── de_funcion1_graficos_semillas/             # Gráficos de convergencia DE para función f1
├── de_funcion2_graficos_semillas/             # Gráficos de convergencia DE para función f2
├── de_funcion3_graficos_semillas/             # Gráficos de convergencia DE para función f3
├── de_funcion4_graficos_semillas/             # Gráficos de convergencia DE para función f4
├── funcion1/                                  # Comparativa de algoritmos para f1 (ACO, BA, DE, GA, PSO)
├── funcion2/                                  # Comparativa de algoritmos para f2
├── funcion3/                                  # Comparativa de algoritmos para f3
├── funcion4/                                  # Comparativa de algoritmos para f4
├── __pycache__/                               # Caché de Python
```


## 📌 Descripción

La tarea consistió en:

- Implementar el algoritmo **DE** desde cero en Python.
- Compararlo preliminarmente con **ACO, GA, PSO y BA** sobre 4 funciones objetivo multimodales.
- Evaluar el rendimiento de DE en 10 configuraciones distintas (combinaciones de NP, F, CR).
- Ejecutar **10 ejecuciones por configuración**, usando **semillas fijas** para reproducibilidad.
- Generar estadísticas y gráficos de convergencia por función y configuración.
- Analizar el efecto de los parámetros en la robustez y desempeño del algoritmo.

## 📈 Resultados

Los resultados completos, incluyendo:
- Rankings por función.
- Gráficos de convergencia de todas las configuraciones.

… se encuentran en:

- `de_output_resumen/resumen_estadisticas_DE_semillas.txt`
- Carpetas `de_funcionX_graficos_semillas/`

## 👥 Autores

- **José Martín Berríos Piña**  
  📧 jose.berrios1@mail.udp.cl

- **Renato Óscar Benjamín Contreras Carvajal**  
  📧 renato.contreras@mail.udp.cl

## 📎 Informe

📄 El documento completo de la tarea se encuentra en:

Informe/Tarea_3_JB_RC.pdf
