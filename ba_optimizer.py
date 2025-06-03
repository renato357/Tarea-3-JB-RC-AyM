# ba_optimizer.py

import math
import random
import numpy as np

# ==============================================================================
# 1. DEFINICIÓN DE FUNCIONES OBJETIVO (Idénticas a aco_optimizer.py)
# ==============================================================================
def f1(variables: list[float]) -> float:
    if len(variables) != 2: raise ValueError("f1 espera 2 variables")
    x1, x2 = variables[0], variables[1]
    return 4 - 4 * x1**3 - 4 * x1 + x2**2

def f2(variables: list[float]) -> float:
    if len(variables) != 6: raise ValueError("f2 espera 6 variables")
    sum_term = sum((variables[i]**2) * (2**(i + 1)) for i in range(6))
    return (1/899) * (sum_term - 1745)

def f3(variables: list[float]) -> float:
    if len(variables) != 2: raise ValueError("f3 espera 2 variables")
    x1, x2 = variables[0], variables[1]
    term1 = x1**6 + x2**4 - 17
    term2 = 2 * x1 + x2 - 4
    return term1**2 + term2**2

def f4(variables: list[float]) -> float:
    if len(variables) != 10: raise ValueError("f4 espera 10 variables")
    sum_log_terms = 0
    product_terms = 1.0
    for xi in variables:
        if not (2 < xi < 10): return float('inf')
        sum_log_terms += (math.log(xi - 2))**2 + (math.log(10 - xi))**2
        product_terms *= xi
    # product_terms será positivo dado 2 < xi < 10
    prod_pow = product_terms**0.2
    return sum_log_terms - prod_pow

# ==============================================================================
# 2. DEFINICIÓN DE FUNCIONES DE RESTRICCIÓN Y LÍMITES (Idénticas a aco_optimizer.py)
# ==============================================================================
def restriccion_f1(variables: list[float]) -> bool:
    x1, x2 = variables[0], variables[1]
    return -5 <= x1 <= 5 and -5 <= x2 <= 5

def limites_dominio_f1() -> list[tuple[float, float]]:
    return [(-5.0, 5.0), (-5.0, 5.0)]

def _verificar_dominio_valor_f2(xi: float) -> bool: return 0 <= xi <= 1
def restriccion_f2(variables: list[float]) -> bool:
    if len(variables) != 6: return False
    return all(_verificar_dominio_valor_f2(xi) for xi in variables)

def limites_dominio_f2() -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * 6

def restriccion_f3(variables: list[float]) -> bool:
    x1, x2 = variables[0], variables[1]
    return -500 <= x1 <= 500 and -500 <= x2 <= 500

def limites_dominio_f3() -> list[tuple[float, float]]:
    return [(-500.0, 500.0), (-500.0, 500.0)]

def _verificar_dominio_valor_f4(xi: float) -> bool: return -2.001 <= xi <= 10
def restriccion_f4(variables: list[float]) -> bool:
    if len(variables) != 10: return False
    return all(_verificar_dominio_valor_f4(xi) for xi in variables)

def limites_dominio_f4_generacion() -> list[tuple[float, float]]: # Para generación y clipping en BA para f4
    epsilon = 1e-6
    return [(2.0 + epsilon, 10.0 - epsilon)] * 10

def limites_dominio_f4_pdf() -> list[tuple[float, float]]: # Límites originales del PDF para f4
    return [(-2.001, 10.0)] * 10


PROBLEMAS = {
    "f1": {"funcion_objetivo": f1, "funcion_restriccion": restriccion_f1, "limites_dominio_pdf": limites_dominio_f1, "limites_generacion": limites_dominio_f1, "dimension": 2},
    "f2": {"funcion_objetivo": f2, "funcion_restriccion": restriccion_f2, "limites_dominio_pdf": limites_dominio_f2, "limites_generacion": limites_dominio_f2, "dimension": 6},
    "f3": {"funcion_objetivo": f3, "funcion_restriccion": restriccion_f3, "limites_dominio_pdf": limites_dominio_f3, "limites_generacion": limites_dominio_f3, "dimension": 2},
    "f4": {"funcion_objetivo": f4, "funcion_restriccion": restriccion_f4, "limites_dominio_pdf": limites_dominio_f4_pdf, "limites_generacion": limites_dominio_f4_generacion, "dimension": 10},
}

# ==============================================================================
# 3. IMPLEMENTACIÓN DEL BAT ALGORITHM (BA)
# ==============================================================================
class BatAlgorithm:
    def __init__(self, nombre_problema: str,
                 num_murcielagos: int, num_iteraciones: int,
                 A_inicial: float, r_inicial_max: float, # r_inicial_max para r_i[0]
                 f_min: float, f_max: float,
                 alpha_loudness: float, gamma_pulserate: float):
        """
        Inicializa el optimizador Bat Algorithm.

        Args:
            nombre_problema (str): Clave del problema en PROBLEMAS.
            num_murcielagos (int): Tamaño de la población de murciélagos.
            num_iteraciones (int): Número máximo de iteraciones.
            A_inicial (float): Loudness inicial (A_0).
            r_inicial_max (float): Tasa de pulso inicial máxima (r_i^0 se sortea hasta este valor).
            f_min (float): Frecuencia mínima.
            f_max (float): Frecuencia máxima.
            alpha_loudness (float): Factor de reducción de loudness (e.g., 0.9-0.99).
            gamma_pulserate (float): Factor de incremento de pulse rate (e.g., 0.1-0.9).
        """
        if nombre_problema not in PROBLEMAS:
            raise ValueError(f"Problema '{nombre_problema}' no definido.")

        self.problema = PROBLEMAS[nombre_problema]
        self.funcion_objetivo = self.problema["funcion_objetivo"]
        self.funcion_restriccion = self.problema["funcion_restriccion"] # Para verificar si la solución está en el dominio del PDF
        self.limites_generacion = self.problema["limites_generacion"]() # Para generar y clipear soluciones
        self.dimension = self.problema["dimension"]

        self.num_murcielagos = num_murcielagos
        self.num_iteraciones = num_iteraciones

        # Parámetros del Bat Algorithm
        self.A_actual = np.full(num_murcielagos, A_inicial) # Loudness A_i para cada murciélago
        self.r_inicial = np.array([random.uniform(0, r_inicial_max) for _ in range(num_murcielagos)]) # r_i^0
        self.r_actual = np.copy(self.r_inicial) # Pulse rate r_i

        self.f_min = f_min
        self.f_max = f_max
        self.alpha_loudness = alpha_loudness # Para A_i = alpha * A_i
        self.gamma_pulserate = gamma_pulserate   # Para r_i = r_0 * (1 - exp(-gamma * t))

        # Población de murciélagos
        self.posiciones = np.zeros((num_murcielagos, self.dimension))
        self.velocidades = np.zeros((num_murcielagos, self.dimension))
        self.fitness = np.full(num_murcielagos, float('inf'))

        # Mejor solución global
        self.mejor_posicion_global = np.zeros(self.dimension)
        self.mejor_fitness_global = float('inf')

        self.historial_convergencia = []

    def _inicializar_poblacion(self):
        for i in range(self.num_murcielagos):
            # Generar posición aleatoria válida
            while True:
                self.posiciones[i] = np.array([random.uniform(self.limites_generacion[d][0], self.limites_generacion[d][1])
                                               for d in range(self.dimension)])
                if self.funcion_restriccion(self.posiciones[i].tolist()): # Verificar contra restricciones del PDF
                    break

            self.velocidades[i] = np.zeros(self.dimension) # Velocidad inicial cero
            self.fitness[i] = self.funcion_objetivo(self.posiciones[i].tolist())

            if self.fitness[i] < self.mejor_fitness_global:
                self.mejor_fitness_global = self.fitness[i]
                self.mejor_posicion_global = np.copy(self.posiciones[i])

        self.historial_convergencia.append(self.mejor_fitness_global)

    def _aplicar_limites(self, posicion: np.ndarray) -> np.ndarray:
        """Aplica los límites de generación a una posición."""
        for d in range(self.dimension):
            posicion[d] = max(self.limites_generacion[d][0], min(posicion[d], self.limites_generacion[d][1]))
        return posicion

    def optimizar(self):
        print(f"Optimizando {self.problema['funcion_objetivo'].__name__} con Bat Algorithm...")
        self._inicializar_poblacion()

        for t in range(self.num_iteraciones):
            media_loudness = np.mean(self.A_actual)

            for i in range(self.num_murcielagos):
                # Actualizar frecuencia, velocidad y posición (fase de exploración global)
                beta = random.random() # Escalar aleatorio [0,1]
                frecuencia_i = self.f_min + (self.f_max - self.f_min) * beta

                self.velocidades[i] += (self.posiciones[i] - self.mejor_posicion_global) * frecuencia_i
                nueva_posicion_i = self.posiciones[i] + self.velocidades[i]
                nueva_posicion_i = self._aplicar_limites(nueva_posicion_i)

                # Búsqueda local (fase de explotación)
                if random.random() > self.r_actual[i]:
                    # Generar solución local alrededor de la mejor solución global
                    epsilon = random.uniform(-1, 1)
                    # Usamos la mejor_posicion_global para la búsqueda local como es común
                    posicion_local = self.mejor_posicion_global + epsilon * media_loudness
                    nueva_posicion_i = self._aplicar_limites(posicion_local)

                # Evaluar la nueva solución
                # Solo evaluar si la nueva posición es válida según las restricciones del PDF
                if self.funcion_restriccion(nueva_posicion_i.tolist()):
                    fitness_nuevo = self.funcion_objetivo(nueva_posicion_i.tolist())
                else:
                    fitness_nuevo = float('inf') # Penalizar si no cumple restricción PDF


                # Aceptar la nueva solución y actualizar loudness y pulse rate
                if random.random() < self.A_actual[i] and fitness_nuevo < self.fitness[i]:
                    self.posiciones[i] = np.copy(nueva_posicion_i)
                    self.fitness[i] = fitness_nuevo

                    # Actualizar Loudness y Pulse Rate
                    self.A_actual[i] *= self.alpha_loudness
                    self.r_actual[i] = self.r_inicial[i] * (1 - math.exp(-self.gamma_pulserate * (t + 1))) # t+1 o t

                # Actualizar la mejor solución global
                if self.fitness[i] < self.mejor_fitness_global:
                    self.mejor_fitness_global = self.fitness[i]
                    self.mejor_posicion_global = np.copy(self.posiciones[i])

            self.historial_convergencia.append(self.mejor_fitness_global)

            if (t + 1) % 10 == 0:
                print(f"Iteración {t + 1}/{self.num_iteraciones} - Mejor Valor: {self.mejor_fitness_global:.6e}")

        print("Optimización con Bat Algorithm completada.")
        print(f"Mejor solución encontrada para {self.problema['funcion_objetivo'].__name__}:")
        print(f"  Variables: {[round(v, 5) for v in self.mejor_posicion_global]}")
        print(f"  Valor Objetivo: {self.mejor_fitness_global:.6e}")
        return self.mejor_posicion_global, self.historial_convergencia


# --- Parámetros del Bat Algorithm (ejemplo, ajustar para cada función y experimento) ---
# Estos son solo ejemplos, debes elegir y justificar 4 configuraciones
CONFIG_BA_DEFAULT = {
    "num_murcielagos": 50,      # Número de murciélagos
    "num_iteraciones": 100,    # Número de iteraciones
    "A_inicial": 0.75,          # Loudness inicial (A_0), e.g., 0.5 - 1.0
    "r_inicial_max": 0.5,       # Tasa de pulso inicial máxima (r^0), e.g., 0.1 - 0.5
    "f_min": 0.0,               # Frecuencia mínima (Q_min)
    "f_max": 2.0,               # Frecuencia máxima (Q_max), e.g., 1.0 - 2.0
    "alpha_loudness": 0.9,      # Factor de reducción de loudness (0 < alpha < 1)
    "gamma_pulserate": 0.9      # Factor de incremento de pulse rate (gamma > 0)
}
# ==============================================================================
# 4. EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":

    # --- Seleccionar el problema a optimizar ---
    nombre_del_problema_a_resolver = "f1" # Ejemplo con f1
    # nombre_del_problema_a_resolver = "f3"
    # nombre_del_problema_a_resolver = "f4"

    print(f"\n--- Resolviendo problema: {nombre_del_problema_a_resolver} con Bat Algorithm ---")

    optimizador_ba = BatAlgorithm(
        nombre_problema=nombre_del_problema_a_resolver,
        num_murcielagos=CONFIG_BA_DEFAULT["num_murcielagos"], # Corregido
        num_iteraciones=CONFIG_BA_DEFAULT["num_iteraciones"], # Corregido
        A_inicial=CONFIG_BA_DEFAULT["A_inicial"], # Corregido
        r_inicial_max=CONFIG_BA_DEFAULT["r_inicial_max"], # Corregido
        f_min=CONFIG_BA_DEFAULT["f_min"], # Corregido
        f_max=CONFIG_BA_DEFAULT["f_max"], # Corregido
        alpha_loudness=CONFIG_BA_DEFAULT["alpha_loudness"], # Corregido
        gamma_pulserate=CONFIG_BA_DEFAULT["gamma_pulserate"] # Corregido
    )

    mejor_posicion, historial = optimizador_ba.optimizar()

    # (Aquí es donde harías más ejecuciones, guardarías resultados, graficarías, etc.
    #  cuando te pida el archivo graficos_ba_optimizer.py)

    # Para tu tarea, recuerda los mismos puntos que con ACO:
    # 1. Realizar 10 ejecuciones por cada función y por cada configuración de parámetros.
    # 2. Usar al menos 4 configuraciones de parámetros distintas, justificándolas.
    # 3. Mostrar resultados, incluyendo gráficos de convergencia.
    # 4. Analizar y concluir sobre los resultados.