# main_graficos.py

import matplotlib.pyplot as plt
import numpy as np
import os # Necesario para crear directorios
import sys # Necesario para redirigir la salida estándar (opcional, para el resumen)

# Importar las clases de los optimizadores y sus configuraciones
try:
    import aco_optimizer as aco
    import ba_optimizer as ba
    import de_optimizer as de
    import ga_optimizer as ga
    import pso_optimizer as pso

    PROBLEMAS = aco.PROBLEMAS # Usar el diccionario PROBLEMAS de uno de los módulos
except ImportError as e:
    print(f"Error importando módulos de optimizadores o sus configuraciones: {e}")
    print("Asegúrate de que todos los archivos *_optimizer.py están en la misma carpeta.")
    exit()

# === 1. DEFINIR 10 CONFIGURACIONES DE PARÁMETROS PARA CADA METAHEURÍSTICA ===
# (Las configuraciones permanecen igual que en el archivo original que me proporcionaste)
CONFIGURACIONES_VARIADAS = {
    "ACO": [
        {"num_hormigas": 30, "num_iteraciones": 100, "tamano_archivo": 10, "q_param_seleccion": 0.2, "xi_param_desviacion": 0.85},
        {"num_hormigas": 15, "num_iteraciones": 100, "tamano_archivo": 5, "q_param_seleccion": 0.1, "xi_param_desviacion": 0.85},
        {"num_hormigas": 50, "num_iteraciones": 100, "tamano_archivo": 15, "q_param_seleccion": 0.3, "xi_param_desviacion": 1.2},
        {"num_hormigas": 30, "num_iteraciones": 100, "tamano_archivo": 5, "q_param_seleccion": 0.8, "xi_param_desviacion": 0.85},
        {"num_hormigas": 30, "num_iteraciones": 100, "tamano_archivo": 20, "q_param_seleccion": 0.2, "xi_param_desviacion": 0.5},
        {"num_hormigas": 20, "num_iteraciones": 100, "tamano_archivo": 7, "q_param_seleccion": 0.15, "xi_param_desviacion": 0.7},
        {"num_hormigas": 40, "num_iteraciones": 100, "tamano_archivo": 12, "q_param_seleccion": 0.4, "xi_param_desviacion": 1.0},
        {"num_hormigas": 30, "num_iteraciones": 100, "tamano_archivo": 10, "q_param_seleccion": 0.05, "xi_param_desviacion": 0.85},
        {"num_hormigas": 30, "num_iteraciones": 100, "tamano_archivo": 10, "q_param_seleccion": 0.2, "xi_param_desviacion": 1.5},
        {"num_hormigas": 25, "num_iteraciones": 100, "tamano_archivo": 8, "q_param_seleccion": 0.25, "xi_param_desviacion": 0.9},
    ],
    "BA": [
        {"num_murcielagos": 50, "num_iteraciones": 100, "A_inicial": 0.75, "r_inicial_max": 0.5, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 25, "num_iteraciones": 100, "A_inicial": 0.9, "r_inicial_max": 0.5, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 70, "num_iteraciones": 100, "A_inicial": 0.75, "r_inicial_max": 0.2, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 50, "num_iteraciones": 100, "A_inicial": 0.75, "r_inicial_max": 0.5, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.95, "gamma_pulserate": 0.5},
        {"num_murcielagos": 50, "num_iteraciones": 100, "A_inicial": 0.75, "r_inicial_max": 0.5, "f_min": 0.5, "f_max": 1.5, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 30, "num_iteraciones": 100, "A_inicial": 0.8, "r_inicial_max": 0.6, "f_min": 0.0, "f_max": 2.5, "alpha_loudness": 0.85, "gamma_pulserate": 0.95},
        {"num_murcielagos": 60, "num_iteraciones": 100, "A_inicial": 0.6, "r_inicial_max": 0.4, "f_min": 0.2, "f_max": 1.8, "alpha_loudness": 0.92, "gamma_pulserate": 0.88},
        {"num_murcielagos": 50, "num_iteraciones": 100, "A_inicial": 0.25, "r_inicial_max": 0.5, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 50, "num_iteraciones": 100, "A_inicial": 0.75, "r_inicial_max": 0.9, "f_min": 0.0, "f_max": 2.0, "alpha_loudness": 0.9, "gamma_pulserate": 0.9},
        {"num_murcielagos": 40, "num_iteraciones": 100, "A_inicial": 0.5, "r_inicial_max": 0.7, "f_min": 0.1, "f_max": 2.2, "alpha_loudness": 0.97, "gamma_pulserate": 0.7},
    ],
    "DE": [
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
    ],
    "GA": [
        {"tamano_poblacion": 100, "prob_crossover": 0.8, "prob_mutacion_gen": 0.1, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.1, "tasa_elitismo": 0.05},
        {"tamano_poblacion": 50, "prob_crossover": 0.8, "prob_mutacion_gen": 0.2, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.15, "tasa_elitismo": 0.05},
        {"tamano_poblacion": 150, "prob_crossover": 0.8, "prob_mutacion_gen": 0.1, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.1, "tasa_elitismo": 0.15},
        {"tamano_poblacion": 100, "prob_crossover": 0.6, "prob_mutacion_gen": 0.1, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.2, "tasa_elitismo": 0.05},
        {"tamano_poblacion": 100, "prob_crossover": 0.8, "prob_mutacion_gen": 0.1, "num_generaciones": 100, "tamano_torneo": 7, "sigma_mutacion": 0.1, "tasa_elitismo": 0.00},
        {"tamano_poblacion": 80, "prob_crossover": 0.7, "prob_mutacion_gen": 0.05, "num_generaciones": 100, "tamano_torneo": 2, "sigma_mutacion": 0.05, "tasa_elitismo": 0.02},
        {"tamano_poblacion": 120, "prob_crossover": 0.9, "prob_mutacion_gen": 0.15, "num_generaciones": 100, "tamano_torneo": 5, "sigma_mutacion": 0.12, "tasa_elitismo": 0.1},
        {"tamano_poblacion": 100, "prob_crossover": 0.8, "prob_mutacion_gen": 0.01, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.02, "tasa_elitismo": 0.05},
        {"tamano_poblacion": 100, "prob_crossover": 0.8, "prob_mutacion_gen": 0.3, "num_generaciones": 100, "tamano_torneo": 3, "sigma_mutacion": 0.3, "tasa_elitismo": 0.05},
        {"tamano_poblacion": 70, "prob_crossover": 0.85, "prob_mutacion_gen": 0.12, "num_generaciones": 100, "tamano_torneo": 4, "sigma_mutacion": 0.08, "tasa_elitismo": 0.08},
    ],
    "PSO": [
        {"num_particulas": 50, "num_iteraciones": 100, "w_inercia": 0.7, "c1_cognitivo": 1.5, "c2_social": 1.5, "limite_velocidad_factor": 0.5},
        {"num_particulas": 25, "num_iteraciones": 100, "w_inercia": 0.4, "c1_cognitivo": 1.5, "c2_social": 1.5, "limite_velocidad_factor": 0.5},
        {"num_particulas": 75, "num_iteraciones": 100, "w_inercia": 0.7, "c1_cognitivo": 2.0, "c2_social": 1.0, "limite_velocidad_factor": 0.5},
        {"num_particulas": 50, "num_iteraciones": 100, "w_inercia": 0.9, "c1_cognitivo": 1.0, "c2_social": 2.0, "limite_velocidad_factor": 0.5},
        {"num_particulas": 50, "num_iteraciones": 100, "w_inercia": 0.7, "c1_cognitivo": 1.5, "c2_social": 1.5, "limite_velocidad_factor": 0.2},
        {"num_particulas": 30, "num_iteraciones": 100, "w_inercia": 0.6, "c1_cognitivo": 1.2, "c2_social": 1.8, "limite_velocidad_factor": 0.6},
        {"num_particulas": 60, "num_iteraciones": 100, "w_inercia": 0.8, "c1_cognitivo": 1.8, "c2_social": 1.2, "limite_velocidad_factor": 0.4},
        {"num_particulas": 50, "num_iteraciones": 100, "w_inercia": 0.2, "c1_cognitivo": 2.5, "c2_social": 2.5, "limite_velocidad_factor": 0.5},
        {"num_particulas": 50, "num_iteraciones": 100, "w_inercia": 0.5, "c1_cognitivo": 1.5, "c2_social": 1.5, "limite_velocidad_factor": 0.5},
        {"num_particulas": 40, "num_iteraciones": 100, "w_inercia": 0.75, "c1_cognitivo": 1.0, "c2_social": 1.0, "limite_velocidad_factor": 0.7},
    ]
}

# Definir los optimizadores y sus clases correspondientes
OPTIMIZADORES_INFO = [
    {"nombre": "ACO", "clase": aco.ACOContinuo, "configs": CONFIGURACIONES_VARIADAS["ACO"]},
    {"nombre": "BA", "clase": ba.BatAlgorithm, "configs": CONFIGURACIONES_VARIADAS["BA"]},
    {"nombre": "DE", "clase": de.DifferentialEvolution, "configs": CONFIGURACIONES_VARIADAS["DE"]},
    {"nombre": "GA", "clase": ga.GeneticAlgorithm, "configs": CONFIGURACIONES_VARIADAS["GA"]},
    {"nombre": "PSO", "clase": pso.ParticleSwarmOptimization, "configs": CONFIGURACIONES_VARIADAS["PSO"]},
]

# Crear directorios si no existen
directorios_problemas = [f"funcion{i+1}" for i in range(len(PROBLEMAS))]
directorio_outputs = "outputs"

for dir_path in directorios_problemas + [directorio_outputs]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directorio creado: {dir_path}")


def format_config_label(config_dict, iter_val_str):
    # Formatea la configuración para la leyenda del gráfico
    label_parts = []
    iter_keys = ["num_iteraciones", "num_generaciones"]

    for key, value in config_dict.items():
        if key in iter_keys or key == "nombre_problema":
            continue
        
        short_key = key.replace("num_", "n_").replace("param_", "p_").replace("seleccion", "sel") \
                       .replace("desviacion", "dev").replace("murcielagos", "bats") \
                       .replace("poblacion", "pop").replace("factor_mutacion_", "F_") \
                       .replace("prob_crossover_", "CR_").replace("prob_mutacion_gen", "Pm") \
                       .replace("tamano_torneo", "tournS").replace("sigma_mutacion", "sigM") \
                       .replace("tasa_elitismo", "elitR").replace("particulas", "parts") \
                       .replace("inercia", "w").replace("cognitivo", "c1").replace("social", "c2") \
                       .replace("limite_velocidad_factor", "vMaxF").replace("_inicial", "_init") \
                       .replace("alpha_loudness", "alphaL").replace("gamma_pulserate", "gammaP") \
                       .replace("tamano_archivo", "archS")
        if isinstance(value, float):
            label_parts.append(f"{short_key}:{value:.2g}")
        else:
            label_parts.append(f"{short_key}:{value}")
    return ", ".join(label_parts)

def ejecutar_experimentos_y_graficar():
    resultados_finales = {}
    resumen_texto = [] # Lista para almacenar el texto del resumen

    for i, nombre_problema in enumerate(PROBLEMAS.keys()):
        resultados_finales[nombre_problema] = {}
        print(f"\n--- PROCESANDO FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---")
        resumen_texto.append(f"\n--- PROCESANDO FUNCIÓN OBJETIVO: {nombre_problema.upper()} ---")

        directorio_destino_grafico = f"funcion{i+1}" # Corresponde a f1 -> funcion1, f2 -> funcion2, etc.

        for meta_info in OPTIMIZADORES_INFO:
            nombre_meta = meta_info["nombre"]
            clase_optimizador = meta_info["clase"]
            configs_meta = meta_info["configs"]
            
            resultados_finales[nombre_problema][nombre_meta] = {
                "mejor_global_fitness": float('inf'),
                "mejor_global_config_idx": -1,
                "mejor_global_config_params": None,
                "corridas_info": []
            }

            print(f"  --- Metaheurística: {nombre_meta} ---")
            resumen_texto.append(f"  --- Metaheurística: {nombre_meta} ---")
            
            plt.figure(figsize=(17, 10))
            
            iter_param_key = "num_iteraciones" if "num_iteraciones" in configs_meta[0] else "num_generaciones"
            iter_val = configs_meta[0].get(iter_param_key, "N/A")
            iter_val_str = f"{iter_val} iter/gen"

            for idx_config, config_params in enumerate(configs_meta):
                print(f"    Ejecutando config {idx_config + 1}/{len(configs_meta)} para {nombre_meta} en {nombre_problema}...")
                
                current_run_config = config_params.copy()
                # Silenciar la salida de optimizador.optimizar() temporalmente si es muy verbosa
                # stdout_original = sys.stdout
                # sys.stdout = open(os.devnull, 'w')

                optimizador = clase_optimizador(nombre_problema=nombre_problema, **current_run_config)
                _, historial_convergencia = optimizador.optimizar() # Asumiendo que optimizar imprime su propio progreso

                # sys.stdout.close()
                # sys.stdout = stdout_original # Restaurar stdout

                if not historial_convergencia:
                    mejor_fitness_esta_corrida = float('inf')
                    iter_mejor_fitness = 0
                    print(f"      ADVERTENCIA: El historial de convergencia está vacío para la config {idx_config + 1}.")
                    resumen_texto.append(f"      ADVERTENCIA: Historial vacío para config {idx_config + 1}.")
                else:
                    mejor_fitness_esta_corrida = min(historial_convergencia)
                    iter_mejor_fitness = np.argmin(historial_convergencia) + 1
                
                run_details = {
                    "config_idx": idx_config,
                    "config_params": current_run_config,
                    "mejor_fitness": mejor_fitness_esta_corrida,
                    "iter_mejor_fitness": iter_mejor_fitness
                }
                resultados_finales[nombre_problema][nombre_meta]["corridas_info"].append(run_details)

                if mejor_fitness_esta_corrida < resultados_finales[nombre_problema][nombre_meta]["mejor_global_fitness"]:
                    resultados_finales[nombre_problema][nombre_meta]["mejor_global_fitness"] = mejor_fitness_esta_corrida
                    resultados_finales[nombre_problema][nombre_meta]["mejor_global_config_idx"] = idx_config
                    resultados_finales[nombre_problema][nombre_meta]["mejor_global_config_params"] = current_run_config
                
                params_label_str = format_config_label(current_run_config, iter_val_str)
                leyenda_label = f"C{idx_config+1}: {params_label_str}\n(Val: {mejor_fitness_esta_corrida:.3e} en iter {iter_mejor_fitness})"
                
                if historial_convergencia:
                    plt.plot(historial_convergencia, label=leyenda_label, linewidth=1.2, alpha=0.8)
                else:
                    plt.plot([0], [float('nan')], label=leyenda_label)


            plt.title(f"Convergencia de {nombre_meta} en {nombre_problema.upper()} ({iter_val_str} por corrida)", fontsize=16)
            plt.xlabel(f"Iteración / Generación", fontsize=12)
            plt.ylabel("Costo de la Función (Mejor Fitness)", fontsize=12)
            plt.yscale('symlog', linthresh=1e-5 if nombre_problema != "f3" else 1e-2) # Ajustar linthresh para f3 si es necesario
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0., fontsize='x-small', frameon=True, facecolor='white', edgecolor='black')
            plt.tight_layout(rect=[0, 0, 0.78, 1])
            
            nombre_archivo_grafico = f"grafico_{nombre_problema}_{nombre_meta}_10configs.png"
            ruta_guardado_grafico = os.path.join(directorio_destino_grafico, nombre_archivo_grafico)
            plt.savefig(ruta_guardado_grafico, bbox_inches='tight', dpi=150)
            print(f"    Gráfico guardado en: {ruta_guardado_grafico}")
            plt.close() # Cerrar la figura para liberar memoria

    # --- 3. RESUMEN DE MEJORES CONFIGURACIONES (ahora se guarda en el archivo de texto) ---
    resumen_texto.append("\n\n" + "="*80)
    resumen_texto.append(" RESUMEN DE MEJORES CONFIGURACIONES POR FUNCIÓN Y METAHEURÍSTICA ")
    resumen_texto.append("="*80)
    for nombre_problema, data_problema in resultados_finales.items():
        resumen_texto.append(f"\nFunción: {nombre_problema.upper()}")
        resumen_texto.append("-"*50)
        for nombre_meta, data_meta in data_problema.items():
            resumen_texto.append(f"  Metaheurística: {nombre_meta}")
            if data_meta["mejor_global_config_idx"] != -1:
                mejor_config_idx = data_meta["mejor_global_config_idx"]
                mejor_fitness = data_meta["mejor_global_fitness"]
                mejor_params_dict = data_meta["mejor_global_config_params"]
                
                params_str_list = []
                iter_val_summary = "N/A"
                iter_key_summary = ""

                for p_name_orig, p_val_orig in mejor_params_dict.items():
                    if p_name_orig in ["num_iteraciones", "num_generaciones"]:
                        iter_key_summary = p_name_orig
                        iter_val_summary = p_val_orig
                        continue
                    if isinstance(p_val_orig, float):
                        params_str_list.append(f"{p_name_orig}={p_val_orig:.2g}")
                    else:
                        params_str_list.append(f"{p_name_orig}={p_val_orig}")
                
                params_str_display = ", ".join(params_str_list)
                if iter_key_summary:
                    params_str_display = f"{iter_key_summary}={iter_val_summary}, {params_str_display}"

                resumen_texto.append(f"    Mejor Configuración (Índice de Corrida: {mejor_config_idx + 1}):")
                resumen_texto.append(f"      Parámetros: {params_str_display}")
                resumen_texto.append(f"      Mejor Fitness Obtenido: {mejor_fitness:.6e}")
            else:
                resumen_texto.append("    No se encontraron resultados válidos para esta metaheurística.")
        resumen_texto.append("="*50)

    # Guardar el resumen en un archivo .txt
    ruta_archivo_resumen = os.path.join(directorio_outputs, "resumen_mejores_configuraciones.txt")
    with open(ruta_archivo_resumen, 'w', encoding='utf-8') as f:
        for linea in resumen_texto:
            f.write(linea + "\n")
            print(linea) # También imprimir en consola para seguimiento
    
    print(f"\nResumen guardado en: {ruta_archivo_resumen}")

if __name__ == "__main__":
    # Para evitar que los optimizadores impriman demasiado y sature la consola,
    # puedes redirigir temporalmente la salida estándar al ejecutar optimizador.optimizar()
    # si los propios optimizadores no tienen una opción de 'verbose=False'.
    # El código actual ya tiene comentada una forma de hacerlo. Si las impresiones de
    # cada optimizador son demasiadas, puedes descomentar esas líneas alrededor de:
    #   _, historial_convergencia = optimizador.optimizar()
    # Sin embargo, los print de "Iteración X/Y - Mejor Valor: Z" dentro de los optimizadores
    # son útiles para el seguimiento, así que considera si realmente quieres silenciarlos.
    # El script imprimirá su propio progreso ("Ejecutando config...", "Gráfico guardado en...").

    ejecutar_experimentos_y_graficar()