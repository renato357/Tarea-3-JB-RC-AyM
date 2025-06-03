# main_graficos.py

import matplotlib.pyplot as plt
# Importamos los componentes necesarios de tu archivo aco_optimizer.py
# Asegúrate de que aco_optimizer.py esté en la misma carpeta.
import aco_optimizer as aco

def ejecutar_y_graficar_todas_las_funciones():
    """
    Ejecuta el optimizador ACO para cada función definida en PROBLEMAS
    y genera un gráfico de convergencia para cada una.
    """
    # --- Parámetros del ACO (puedes usar los de tu tarea o ajustarlos) ---
    # Para la tarea completa, recuerda que debes probar al menos 4 configuraciones distintas[cite: 8].
    # Esta es solo una configuración de ejemplo.
    configuracion_actual_aco = {
        "num_hormigas": 30,
        "num_iteraciones": 1000, # Aumentado como en tu último código
        "tamano_archivo": 10,
        "q_param_seleccion": 0.2,
        "xi_param_desviacion": 0.85
    }

    # Iteramos sobre todas las funciones definidas en el diccionario PROBLEMAS de aco_optimizer.py
    for nombre_problema in aco.PROBLEMAS.keys():
        print(f"\n--- Ejecutando optimización para la función: {nombre_problema} ---")

        # Crear instancia del optimizador ACO para el problema actual
        optimizador = aco.ACOContinuo(
            nombre_problema=nombre_problema,
            num_hormigas=configuracion_actual_aco["num_hormigas"],
            num_iteraciones=configuracion_actual_aco["num_iteraciones"],
            tamano_archivo=configuracion_actual_aco["tamano_archivo"],
            q_param_seleccion=configuracion_actual_aco["q_param_seleccion"],
            xi_param_desviacion=configuracion_actual_aco["xi_param_desviacion"]
        )

        # Ejecutar la optimización
        # La función optimizar() ya imprime la mejor solución encontrada.
        mejor_solucion, historial_convergencia = optimizador.optimizar()

        # --- Generar el gráfico de convergencia ---
        # Tu tarea requiere gráficos de convergencia para cada configuración de parámetros[cite: 9].
        plt.figure(figsize=(10, 6)) # Tamaño del gráfico
        plt.plot(historial_convergencia, label=f"Mejor Valor Objetivo ({nombre_problema})")
        plt.title(f"Convergencia de ACO para {nombre_problema}\nConfig: {configuracion_actual_aco}")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor Valor Objetivo")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        # Guardar el gráfico (opcional, pero útil para el informe)
        # plt.savefig(f"convergencia_{nombre_problema}.png")
        
        plt.show() # Muestra el gráfico

    print("\n--- Todas las funciones han sido procesadas y graficadas. ---")

if __name__ == "__main__":
    ejecutar_y_graficar_todas_las_funciones()