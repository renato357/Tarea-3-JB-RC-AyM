# graficos_ga_optimizer.py

import matplotlib.pyplot as plt
# Importamos los componentes necesarios de tu archivo ga_optimizer.py
# Asegúrate de que ga_optimizer.py esté en la misma carpeta.
import ga_optimizer as ga # 'ga' será el alias para acceder a tu código

def ejecutar_ga_y_graficar_funciones():
    """
    Ejecuta el Genetic Algorithm para cada función definida en PROBLEMAS
    y genera un gráfico de convergencia para cada una.
    """
    # --- Parámetros del Genetic Algorithm (puedes usar los de tu tarea o ajustarlos) ---
    # Para la tarea completa, recuerda que debes probar al menos 4 configuraciones distintas.
    # Esta es solo una configuración de ejemplo.
    configuracion_ga_actual = {
        "tamano_poblacion": 100,
        "prob_crossover": 0.8,
        "prob_mutacion_gen": 0.1,
        "num_generaciones": 1000, # Según el ejemplo en ga_optimizer.py
        "tamano_torneo": 3,
        "sigma_mutacion": 0.1,
        "tasa_elitismo": 0.05
    }

    # Iteramos sobre todas las funciones definidas en el diccionario PROBLEMAS de ga_optimizer.py
    for nombre_problema in ga.PROBLEMAS.keys():
        print(f"\n--- Ejecutando Genetic Algorithm para la función: {nombre_problema} ---")

        # Crear instancia del optimizador Genetic Algorithm para el problema actual
        optimizador = ga.GeneticAlgorithm(
            nombre_problema=nombre_problema,
            tamano_poblacion=configuracion_ga_actual["tamano_poblacion"],
            prob_crossover=configuracion_ga_actual["prob_crossover"],
            prob_mutacion_gen=configuracion_ga_actual["prob_mutacion_gen"],
            num_generaciones=configuracion_ga_actual["num_generaciones"],
            tamano_torneo=configuracion_ga_actual["tamano_torneo"],
            sigma_mutacion=configuracion_ga_actual["sigma_mutacion"],
            tasa_elitismo=configuracion_ga_actual["tasa_elitismo"]
        )

        # Ejecutar la optimización
        # La función optimizar() ya imprime la mejor solución encontrada.
        mejor_individuo, historial_convergencia = optimizador.optimizar()

        # --- Generar el gráfico de convergencia ---
        # Tu tarea requiere gráficos de convergencia para cada configuración de parámetros.
        plt.figure(figsize=(10, 6)) # Tamaño del gráfico
        plt.plot(historial_convergencia, label=f"Mejor Fitness ({nombre_problema})")
        
        # Crear un string legible de la configuración para el título
        config_str_list = [f"{k}={v}" for k, v in configuracion_ga_actual.items()]
        # Dividir el string de configuración en varias líneas si es muy largo
        max_items_por_linea = 3 # Ajusta según sea necesario
        config_display_lines = []
        for i in range(0, len(config_str_list), max_items_por_linea):
            config_display_lines.append(", ".join(config_str_list[i:i+max_items_por_linea]))
        config_display_str = "\n".join(config_display_lines)
        
        plt.title(f"Convergencia de GA para {nombre_problema}\nConfig: {config_display_str}", fontsize=10)
        plt.xlabel("Generación")
        plt.ylabel("Mejor Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        # Guardar el gráfico (opcional, pero útil para el informe)
        # plt.savefig(f"convergencia_ga_{nombre_problema}.png")
        
        plt.show() # Muestra el gráfico

    print("\n--- Todas las funciones han sido procesadas y graficadas con Genetic Algorithm. ---")

if __name__ == "__main__":
    ejecutar_ga_y_graficar_funciones()