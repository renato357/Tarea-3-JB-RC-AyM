# graficos_de_optimizer.py

import matplotlib.pyplot as plt
# Importamos los componentes necesarios de tu archivo de_optimizer.py
# Asegúrate de que de_optimizer.py esté en la misma carpeta.
import de_optimizer as de # 'de' será el alias para acceder a tu código

def ejecutar_de_y_graficar_funciones():
    """
    Ejecuta Differential Evolution para cada función definida en PROBLEMAS
    y genera un gráfico de convergencia para cada una.
    """
    # --- Parámetros de Differential Evolution (puedes usar los de tu tarea o ajustarlos) ---
    # Para la tarea completa, recuerda que debes probar al menos 4 configuraciones distintas.
    # Esta es solo una configuración de ejemplo.
    configuracion_de_actual = {
        "tamano_poblacion": 50,
        "factor_mutacion_F": 0.7,
        "prob_crossover_CR": 0.8,
        "num_generaciones": 1000  # Según el ejemplo en de_optimizer.py
    }

    # Iteramos sobre todas las funciones definidas en el diccionario PROBLEMAS de de_optimizer.py
    for nombre_problema in de.PROBLEMAS.keys():
        print(f"\n--- Ejecutando Differential Evolution para la función: {nombre_problema} ---")

        # Crear instancia del optimizador Differential Evolution para el problema actual
        optimizador = de.DifferentialEvolution(
            nombre_problema=nombre_problema,
            tamano_poblacion=configuracion_de_actual["tamano_poblacion"],
            factor_mutacion_F=configuracion_de_actual["factor_mutacion_F"],
            prob_crossover_CR=configuracion_de_actual["prob_crossover_CR"],
            num_generaciones=configuracion_de_actual["num_generaciones"]
            # estrategia_mutacion se toma por defecto ("rand/1")
        )

        # Ejecutar la optimización
        # La función optimizar() ya imprime la mejor solución encontrada.
        mejor_individuo, historial_convergencia = optimizador.optimizar()

        # --- Generar el gráfico de convergencia ---
        # Tu tarea requiere gráficos de convergencia para cada configuración de parámetros.
        plt.figure(figsize=(10, 6)) # Tamaño del gráfico
        plt.plot(historial_convergencia, label=f"Mejor Fitness ({nombre_problema})")
        
        # Crear un string legible de la configuración para el título
        config_str_list = [f"{k}={v}" for k, v in configuracion_de_actual.items()]
        # Dividir el string de configuración en dos líneas si es muy largo
        if len(config_str_list) > 2:
            config_str_line1 = ", ".join(config_str_list[:len(config_str_list)//2])
            config_str_line2 = ", ".join(config_str_list[len(config_str_list)//2:])
            config_display_str = f"{config_str_line1},\n{config_str_line2}"
        else:
            config_display_str = ", ".join(config_str_list)

        plt.title(f"Convergencia de DE para {nombre_problema}\nConfig: {config_display_str}", fontsize=10)
        plt.xlabel("Generación")
        plt.ylabel("Mejor Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        # Guardar el gráfico (opcional, pero útil para el informe)
        # plt.savefig(f"convergencia_de_{nombre_problema}.png")
        
        plt.show() # Muestra el gráfico

    print("\n--- Todas las funciones han sido procesadas y graficadas con Differential Evolution. ---")

if __name__ == "__main__":
    ejecutar_de_y_graficar_funciones()