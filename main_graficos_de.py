# main_graficos_de.py

import matplotlib.pyplot as plt
import numpy as np
import os
import time
# import random # No es necesario importar random aquí si DE lo maneja internamente

# Importar la clase del optimizador DE y sus definiciones de problemas
try:
    import de_optimizer as de
    PROBLEMAS = de.PROBLEMAS 
except ImportError as e:
    print(f"Error importando el módulo de_optimizer.py: {e}")
    print("Asegúrate de que el archivo de_optimizer.py está en la misma carpeta.")
    exit()

# Configuraciones variadas para DE
CONFIGURACIONES_DE_VARIADAS = [
    {"tamano_poblacion": 50, "factor_mutacion_F": 0.7, "prob_crossover_CR": 0.8, "num_generaciones": 100},
    {"tamano_poblacion": 25, "factor_mutacion_F": 0.5, "prob_crossover_CR": 0.8, "num_generaciones": 100},
    {"tamano_poblacion": 75, "factor_mutacion_F": 0.7, "prob_crossover_CR": 0.95, "num_generaciones": 100},
    {"tamano_poblacion": 50, "factor_mutacion_F": 0.9, "prob_crossover_CR": 0.8, "num_generaciones": 100},
    {"tamano_poblacion": 50, "factor_mutacion_F": 0.7, "prob_crossover_CR": 0.5, "num_generaciones": 100},
    {"tamano_poblacion": 30, "factor_mutacion_F": 0.6, "prob_crossover_CR": 0.7, "num_generaciones": 100},
    {"tamano_poblacion": 60, "factor_mutacion_F": 0.8, "prob_crossover_CR": 0.9, "num_generaciones": 100},
    {"tamano_poblacion": 50, "factor_mutacion_F": 0.4, "prob_crossover_CR": 0.4, "num_generaciones": 100},
    {"tamano_poblacion": 50, "factor_mutacion_F": 1.0, "prob_crossover_CR": 1.0, "num_generaciones": 100},
    {"tamano_poblacion": 40, "factor_mutacion_F": 0.75, "prob_crossover_CR": 0.85, "num_generaciones": 100},
]

NUM_EJECUCIONES_POR_CONFIG = 10

# MODIFICADO: Definir una lista de semillas a utilizar para las ejecuciones
# Asegurarse que la longitud de esta lista sea igual a NUM_EJECUCIONES_POR_CONFIG
LISTA_SEMILLAS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] 
# (Puedes cambiar estas semillas por las que prefieras)

if len(LISTA_SEMILLAS) != NUM_EJECUCIONES_POR_CONFIG:
    # Este error detendrá la ejecución si las longitudes no coinciden.
    raise ValueError(
        f"La longitud de LISTA_SEMILLAS ({len(LISTA_SEMILLAS)}) debe ser igual a "
        f"NUM_EJECUCIONES_POR_CONFIG ({NUM_EJECUCIONES_POR_CONFIG})."
    )

# --- 1. CREACIÓN DE DIRECTORIOS ---
print("Creando directorios para los resultados de DE...")
directorios_base_graficos = []
if not PROBLEMAS:
    print("Error: El diccionario PROBLEMAS está vacío o no se cargó correctamente desde de_optimizer.py.")
    exit()

for i in range(len(PROBLEMAS)):
    dir_name = f"de_funcion{i+1}_graficos_semillas" # MODIFICADO: Nombre de directorio
    directorios_base_graficos.append(dir_name)
    os.makedirs(dir_name, exist_ok=True)

directorio_output_resumen = "de_output_resumen" # El resumen puede ir al mismo directorio general
os.makedirs(directorio_output_resumen, exist_ok=True)
print("Directorios creados/verificados.\n")


def format_params_for_title(params_dict, include_gens=False):
    parts = []
    for key, value in params_dict.items():
        if not include_gens and key == "num_generaciones": continue
        short_key = key.replace("tamano_poblacion", "NP") \
                       .replace("factor_mutacion_F", "F") \
                       .replace("prob_crossover_CR", "CR") \
                       .replace("num_generaciones", "Gens")
        if isinstance(value, float):
            parts.append(f"{short_key}:{value:.2g}")
        else:
            parts.append(f"{short_key}:{value}")
    return ", ".join(parts)

# --- 2. EJECUCIÓN DE EXPERIMENTOS Y COLECCIÓN DE DATOS ---
resultados_completos = {} 
tiempo_inicio_total = time.time()

for i_problema, (nombre_problema, info_problema) in enumerate(PROBLEMAS.items()):
    print(f"--- PROCESANDO FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---")
    resultados_completos[nombre_problema] = {}
    directorio_graficos_actual = directorios_base_graficos[i_problema]

    for idx_config, config_params_de in enumerate(CONFIGURACIONES_DE_VARIADAS):
        print(f"  Configuración DE {idx_config + 1}/{len(CONFIGURACIONES_DE_VARIADAS)}: {format_params_for_title(config_params_de)}")
        
        resultados_completos[nombre_problema][idx_config] = {
            "parametros": config_params_de,
            "fitness_finales_ejecuciones": [],
            "historiales_convergencia_ejecuciones": [],
            "semillas_usadas_ejecuciones": [] # MODIFICADO: Para guardar las semillas usadas
        }

        plt.figure(figsize=(12, 7)) # Tamaño del gráfico
        iter_generaciones = config_params_de.get("num_generaciones", 100)

        for ejecucion in range(NUM_EJECUCIONES_POR_CONFIG):
            # MODIFICADO: Obtener y usar la semilla para esta ejecución
            semilla_actual = LISTA_SEMILLAS[ejecucion]
            print(f"    Ejecución {ejecucion + 1}/{NUM_EJECUCIONES_POR_CONFIG} (Semilla: {semilla_actual})...")
            
            optimizador = de.DifferentialEvolution(
                nombre_problema=nombre_problema,
                semilla=semilla_actual, # MODIFICADO: Pasar la semilla al optimizador
                **config_params_de
            )
            
            _, historial_convergencia = optimizador.optimizar()

            # Guardar la semilla usada para esta ejecución específica
            resultados_completos[nombre_problema][idx_config]["semillas_usadas_ejecuciones"].append(semilla_actual)

            if historial_convergencia and len(historial_convergencia) > 0 : # Asegurarse que el historial no esté vacío
                fitness_final = historial_convergencia[-1]
                resultados_completos[nombre_problema][idx_config]["fitness_finales_ejecuciones"].append(fitness_final)
                resultados_completos[nombre_problema][idx_config]["historiales_convergencia_ejecuciones"].append(historial_convergencia)
                # MODIFICADO: Añadir semilla (S:{semilla_actual}) a la etiqueta de la leyenda
                plt.plot(historial_convergencia, label=f"Eje {ejecucion + 1} (S:{semilla_actual}, F:{fitness_final:.3e})", alpha=0.7)
            else:
                # Si el historial está vacío o tiene problemas, registrar un valor alto y un historial vacío/NaNs
                resultados_completos[nombre_problema][idx_config]["fitness_finales_ejecuciones"].append(float('inf'))
                # Crear un historial de 'inf' si está vacío, para mantener consistencia en la longitud de los datos (+1 por el estado inicial)
                hist_vacio_o_fallido = [float('inf')] * (iter_generaciones + 1) 
                resultados_completos[nombre_problema][idx_config]["historiales_convergencia_ejecuciones"].append(hist_vacio_o_fallido)
                print(f"      ADVERTENCIA: Historial vacío o inválido para {nombre_problema}, config {idx_config+1}, ejecución {ejecucion+1} (S:{semilla_actual}). Fitness registrado como Inf.")
                # Opcional: plotear una línea plana en 'inf' o no plotear nada para ejecuciones fallidas
                # plt.plot(hist_vacio_o_fallido, label=f"Eje {ejecucion + 1} (S:{semilla_actual}, Fallida)", alpha=0.4, linestyle=':')


        params_str_titulo = format_params_for_title(config_params_de)
        # MODIFICADO: Título del gráfico puede mencionar el uso de semillas fijas.
        plt.title(f"Convergencia DE para {nombre_problema.upper()} - Config {idx_config + 1}\nParams: {params_str_titulo}\n({NUM_EJECUCIONES_POR_CONFIG} ejecuciones con semillas fijas)")        
        plt.xlabel("Generación")
        plt.ylabel("Mejor Fitness")
        
        # Lógica para escala logarítmica mejorada
        linthresh_val = 1e-5
        if nombre_problema.lower() == "f3": 
            linthresh_val = 1e-1 
        elif nombre_problema.lower() == "f4": 
             linthresh_val = 1e-10
        
        # Verificar si hay datos válidos para la escala symlog
        all_finite_hist_values = []
        for hist_list in resultados_completos[nombre_problema][idx_config]["historiales_convergencia_ejecuciones"]:
            for val in hist_list:
                if val is not None and val != float('inf') and val != -float('inf'):
                    all_finite_hist_values.append(val)
        
        if any(v != 0 for v in all_finite_hist_values): # Aplicar symlog si hay valores finitos no cero
            try:
                plt.yscale('symlog', linthresh=linthresh_val, linscale=0.1) # linscale puede ayudar con la visualización cerca de cero
            except ValueError as e_yscale: 
                print(f"Advertencia: No se pudo aplicar escala symlog para {nombre_problema} config {idx_config+1}: {e_yscale}. Usando escala lineal.")
                plt.yscale('linear')
        else: # Si todos los valores son 0, inf, o no hay datos válidos, usar lineal.
            plt.yscale('linear')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize='small', loc='upper right') # Ajustar loc o usar bbox_to_anchor si la leyenda es muy grande
        plt.tight_layout() # Ajusta automáticamente los parámetros del subplot para dar un buen ajuste.

        nombre_archivo_grafico = f"DE_{nombre_problema}_config{idx_config+1}_semillas.png" # MODIFICADO: Nombre de archivo
        ruta_guardado_grafico = os.path.join(directorio_graficos_actual, nombre_archivo_grafico)
        plt.savefig(ruta_guardado_grafico, dpi=150)
        plt.close() # Cerrar la figura para liberar memoria
        print(f"    Gráfico guardado en: {ruta_guardado_grafico}")
    print(f"--- FIN PROCESAMIENTO FUNCIÓN: {nombre_problema.upper()} ---\n")

tiempo_fin_total = time.time()
print(f"Tiempo total de ejecución de experimentos: {tiempo_fin_total - tiempo_inicio_total:.2f} segundos.\n")

# --- 3. GENERACIÓN DEL ARCHIVO DE RESUMEN ---
print("Generando archivo de resumen de estadísticas para DE...")
resumen_texto_final = []
resumen_texto_final.append("="*80)
resumen_texto_final.append(" RESUMEN DE ESTADÍSTICAS PARA EVOLUCIÓN DIFERENCIAL (DE) ")
resumen_texto_final.append(f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}")
resumen_texto_final.append(f"Número de ejecuciones por configuración: {NUM_EJECUCIONES_POR_CONFIG}")
# MODIFICADO: Añadir información sobre las semillas al resumen
resumen_texto_final.append(f"Semillas utilizadas para las ejecuciones (repetidas para cada config/función): {LISTA_SEMILLAS}")
resumen_texto_final.append("="*80)

for nombre_problema, datos_configs_problema in resultados_completos.items():
    resumen_texto_final.append(f"\n\n--- FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---")
    
    for idx_config, data_config_especifica in datos_configs_problema.items():
        parametros_usados = data_config_especifica["parametros"]
        fitness_obtenidos = data_config_especifica["fitness_finales_ejecuciones"]
        # semillas_config = data_config_especifica["semillas_usadas_ejecuciones"] # Disponible si se necesita mostrar por config

        resumen_texto_final.append(f"\n  Configuración DE {idx_config + 1}:")
        resumen_texto_final.append(f"    Parámetros: {format_params_for_title(parametros_usados, include_gens=True)}")

        if fitness_obtenidos and not all(f == float('inf') for f in fitness_obtenidos):
            # Filtrar 'inf' para cálculos estadísticos si algunos resultados fueron válidos
            fitness_validos = [f for f in fitness_obtenidos if f != float('inf')]
            if not fitness_validos: # Si todos fueron 'inf' después de filtrar (aunque la lista original no era toda inf)
                 resumen_texto_final.append("    Todas las ejecuciones consideradas válidas resultaron en fitness infinito.")
            else:
                mejor_fitness_global_config = np.min(fitness_validos)
                peor_fitness_global_config = np.max(fitness_validos)
                promedio_fitness_global_config = np.mean(fitness_validos)
                std_dev_fitness_config = np.std(fitness_validos)
                mediana_fitness_config = np.median(fitness_validos)

                resumen_texto_final.append(f"    Resultados de las {len(fitness_validos)} ejecuciones válidas (de {len(fitness_obtenidos)} totales):")
                resumen_texto_final.append(f"      Mejor Fitness (Mínimo):   {mejor_fitness_global_config:.6e}")
                resumen_texto_final.append(f"      Peor Fitness (Máximo):    {peor_fitness_global_config:.6e}")
                resumen_texto_final.append(f"      Promedio Fitness:         {promedio_fitness_global_config:.6e}")
                resumen_texto_final.append(f"      Mediana Fitness:          {mediana_fitness_config:.6e}")
                resumen_texto_final.append(f"      Desviación Estándar:      {std_dev_fitness_config:.6e}")
        else:
            resumen_texto_final.append("    No se obtuvieron resultados de fitness válidos (o todos fueron Inf) para esta configuración.")
    resumen_texto_final.append("-"*(len(f"--- FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---") + 4 ))


# --- 4. NUEVA SECCIÓN: RANKING DE CONFIGURACIONES POR FUNCIÓN ---
resumen_texto_final.append("\n\n" + "="*80)
resumen_texto_final.append(" RANKING DE MEJORES CONFIGURACIONES POR FUNCIÓN OBJETIVO (DE) ")
resumen_texto_final.append(" (Basado en el Mejor Fitness (Mínimo) obtenido en las ejecuciones con semillas fijas) ")
resumen_texto_final.append("="*80)

for nombre_problema, datos_configs_problema in resultados_completos.items():
    resumen_texto_final.append(f"\n\n--- FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---")
    
    config_data_for_ranking = []
    for idx_config, data_config_especifica in datos_configs_problema.items():
        parametros_usados = data_config_especifica["parametros"]
        fitness_obtenidos = data_config_especifica["fitness_finales_ejecuciones"]
        
        params_str_ranking = format_params_for_title(parametros_usados, include_gens=True)
        config_display_name = f"Config {idx_config + 1} ({params_str_ranking})"

        if fitness_obtenidos and not all(f == float('inf') for f in fitness_obtenidos):
            fitness_validos = [f for f in fitness_obtenidos if f != float('inf')]
            if fitness_validos: # Solo si hay al menos un fitness válido
                 mejor_fitness_config = np.min(fitness_validos)
                 config_data_for_ranking.append({
                    "nombre_display": config_display_name,
                    "mejor_fitness": mejor_fitness_config,
                    "parametros_obj": parametros_usados 
                })
            else: # Todos fueron inf, pero la lista fitness_obtenidos no estaba vacía inicialmente
                config_data_for_ranking.append({
                    "nombre_display": config_display_name,
                    "mejor_fitness": float('inf'), # Marcar como inf si no hay válidos
                    "parametros_obj": parametros_usados
                })
        else: # No hubo resultados o todos fueron inf desde el inicio
            config_data_for_ranking.append({
                "nombre_display": config_display_name,
                "mejor_fitness": float('inf'),
                "parametros_obj": parametros_usados
            })

    # Ordenar las configuraciones por 'mejor_fitness' (ascendente)
    sorted_config_data = sorted(config_data_for_ranking, key=lambda x: x['mejor_fitness'])

    resumen_texto_final.append("  Ranking de configuraciones:")
    
    current_rank = 0
    last_fitness_val = -float('inf') # Para manejar el primer elemento y los empates
    items_processed_for_rank = 0 # Contador para asignar rangos

    if not sorted_config_data:
        resumen_texto_final.append("    No hay datos de configuración para rankear.")
    else:
        for item_data in sorted_config_data:
            items_processed_for_rank += 1
            # Actualizar el rango si el fitness actual es diferente al anterior (mejor es menor)
            # Usar np.isclose para comparación de flotantes
            if not np.isclose(item_data['mejor_fitness'], last_fitness_val):
                current_rank = items_processed_for_rank
            
            if item_data['mejor_fitness'] == float('inf'):
                fitness_display = "Inf (No válido)"
            else:
                fitness_display = f"{item_data['mejor_fitness']:.6e}"

            resumen_texto_final.append(f"    Rango {current_rank}: {item_data['nombre_display']} - Mejor Fitness: {fitness_display}")
            last_fitness_val = item_data['mejor_fitness'] # Actualizar el último fitness visto
            
    resumen_texto_final.append("-"*(len(f"--- FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---") +4))


# --- 5. GUARDAR EL ARCHIVO DE RESUMEN COMPLETO ---
nombre_archivo_resumen = "resumen_estadisticas_DE_semillas.txt" # MODIFICADO: Nombre de archivo
ruta_archivo_resumen = os.path.join(directorio_output_resumen, nombre_archivo_resumen)

print(f"\nGuardando resumen completo en: {ruta_archivo_resumen}")
with open(ruta_archivo_resumen, 'w', encoding='utf-8') as f:
    for linea in resumen_texto_final:
        f.write(linea + "\n")
        # print(linea) # Opcional: imprimir también en consola

print(f"\nResumen de estadísticas y ranking para DE (con semillas) guardado en: {ruta_archivo_resumen}")
print("Proceso completado.")

if __name__ == "__main__":
    # La ejecución se inicia directamente cuando se corre el script.
    # Asegúrate de que `de_optimizer.py` (modificado) está en el mismo directorio.
    pass
