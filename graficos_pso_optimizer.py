# graficos_pso_optimizer.py

import matplotlib.pyplot as plt
# Importamos los componentes necesarios de tu archivo pso_optimizer.py
# Asegúrate de que pso_optimizer.py esté en la misma carpeta.
import pso_optimizer as pso # 'pso' será el alias para acceder a tu código

def ejecutar_pso_y_graficar_funciones():
    """
    Ejecuta Particle Swarm Optimization para cada función definida en PROBLEMAS
    y genera un gráfico de convergencia para cada una.
    """
    # --- Parámetros de Particle Swarm Optimization (puedes usar los de tu tarea o ajustarlos) ---
    # Para la tarea completa, recuerda que debes probar al menos 4 configuraciones distintas.
    # Esta es solo una configuración de ejemplo.
    configuracion_pso_actual = {
        "num_particulas": 50,
        "num_iteraciones": 1000, # Según el ejemplo en pso_optimizer.py
        "w_inercia": 0.7,
        "c1_cognitivo": 1.5,
        "c2_social": 1.5,
        "limite_velocidad_factor": 0.5
    }

    # Iteramos sobre todas las funciones definidas en el diccionario PROBLEMAS de pso_optimizer.py
    for nombre_problema in pso.PROBLEMAS.keys():
        print(f"\n--- Ejecutando Particle Swarm Optimization para la función: {nombre_problema} ---")

        # Crear instancia del optimizador Particle Swarm Optimization para el problema actual
        optimizador = pso.ParticleSwarmOptimization(
            nombre_problema=nombre_problema,
            num_particulas=configuracion_pso_actual["num_particulas"],
            num_iteraciones=configuracion_pso_actual["num_iteraciones"],
            w_inercia=configuracion_pso_actual["w_inercia"],
            c1_cognitivo=configuracion_pso_actual["c1_cognitivo"],
            c2_social=configuracion_pso_actual["c2_social"],
            limite_velocidad_factor=configuracion_pso_actual["limite_velocidad_factor"]
        )

        # Ejecutar la optimización
        # La función optimizar() ya imprime la mejor solución encontrada.
        mejor_posicion, historial_convergencia = optimizador.optimizar()

        # --- Generar el gráfico de convergencia ---
        # Tu tarea requiere gráficos de convergencia para cada configuración de parámetros.
        plt.figure(figsize=(10, 6)) # Tamaño del gráfico
        plt.plot(historial_convergencia, label=f"Mejor Fitness Global ({nombre_problema})")
        
        # Crear un string legible de la configuración para el título
        config_str_list = [f"{k}={v}" for k, v in configuracion_pso_actual.items()]
        # Dividir el string de configuración en varias líneas si es muy largo
        max_items_por_linea = 3 # Ajusta según sea necesario
        config_display_lines = []
        for i in range(0, len(config_str_list), max_items_por_linea):
            config_display_lines.append(", ".join(config_str_list[i:i+max_items_por_linea]))
        config_display_str = "\n".join(config_display_lines)
        
        plt.title(f"Convergencia de PSO para {nombre_problema}\nConfig: {config_display_str}", fontsize=10)
        plt.xlabel("Iteración")
        plt.ylabel("Mejor Fitness Global")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        # Guardar el gráfico (opcional, pero útil para el informe)
        # plt.savefig(f"convergencia_pso_{nombre_problema}.png")
        
        plt.show() # Muestra el gráfico

    print("\n--- Todas las funciones han sido procesadas y graficadas con Particle Swarm Optimization. ---")

if __name__ == "__main__":
    ejecutar_pso_y_graficar_funciones()