# graficos_ba_optimizer.py

import matplotlib.pyplot as plt
# Importamos los componentes necesarios de tu archivo ba_optimizer.py
# Asegúrate de que ba_optimizer.py esté en la misma carpeta.
import ba_optimizer as ba # 'ba' será el alias para acceder a tu código

def ejecutar_ba_y_graficar_funciones():
    """
    Ejecuta el Bat Algorithm para cada función definida en PROBLEMAS
    y genera un gráfico de convergencia para cada una.
    """
    # --- Parámetros del Bat Algorithm (puedes usar los de tu tarea o ajustarlos) ---
    # Para la tarea completa[cite: 7, 8], recuerda que debes probar al menos 4 configuraciones distintas.
    # Esta es solo una configuración de ejemplo.
    configuracion_ba_actual = {
        "num_murcielagos": 40,
        "num_iteraciones": 1000,    # Según el ejemplo en ba_optimizer.py
        "A_inicial": 0.75,
        "r_inicial_max": 0.5,
        "f_min": 0.0,
        "f_max": 2.0,
        "alpha_loudness": 0.9,
        "gamma_pulserate": 0.9
    }

    # Iteramos sobre todas las funciones definidas en el diccionario PROBLEMAS de ba_optimizer.py
    for nombre_problema in ba.PROBLEMAS.keys():
        print(f"\n--- Ejecutando Bat Algorithm para la función: {nombre_problema} ---")

        # Crear instancia del optimizador Bat Algorithm para el problema actual
        optimizador = ba.BatAlgorithm(
            nombre_problema=nombre_problema,
            num_murcielagos=configuracion_ba_actual["num_murcielagos"],
            num_iteraciones=configuracion_ba_actual["num_iteraciones"],
            A_inicial=configuracion_ba_actual["A_inicial"],
            r_inicial_max=configuracion_ba_actual["r_inicial_max"],
            f_min=configuracion_ba_actual["f_min"],
            f_max=configuracion_ba_actual["f_max"],
            alpha_loudness=configuracion_ba_actual["alpha_loudness"],
            gamma_pulserate=configuracion_ba_actual["gamma_pulserate"]
        )

        # Ejecutar la optimización
        # La función optimizar() ya imprime la mejor solución encontrada.
        mejor_posicion, historial_convergencia = optimizador.optimizar()

        # --- Generar el gráfico de convergencia ---
        # Tu tarea requiere gráficos de convergencia para cada configuración de parámetros[cite: 9].
        plt.figure(figsize=(10, 6)) # Tamaño del gráfico
        plt.plot(historial_convergencia, label=f"Mejor Valor Objetivo ({nombre_problema})")
        # Crear un string legible de la configuración para el título
        config_str = ", ".join([f"{k}={v}" for k, v in configuracion_ba_actual.items()])
        plt.title(f"Convergencia de Bat Algorithm para {nombre_problema}\nConfig: {config_str}", fontsize=10)
        plt.xlabel("Iteración")
        plt.ylabel("Mejor Valor Objetivo")
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        # Guardar el gráfico (opcional, pero útil para el informe)
        # plt.savefig(f"convergencia_ba_{nombre_problema}.png")
        
        plt.show() # Muestra el gráfico

    print("\n--- Todas las funciones han sido procesadas y graficadas con Bat Algorithm. ---")

if __name__ == "__main__":
    ejecutar_ba_y_graficar_funciones()