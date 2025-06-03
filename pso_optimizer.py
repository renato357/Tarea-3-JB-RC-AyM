# pso_optimizer.py

import math
import random
import numpy as np

# ==============================================================================
# 1. DEFINICIÓN DE FUNCIONES OBJETIVO (Idénticas a implementaciones anteriores)
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
        if not (2 < xi < 10): return float('inf') # Dominio estricto para ln
        sum_log_terms += (math.log(xi - 2))**2 + (math.log(10 - xi))**2
        product_terms *= xi
    prod_pow = product_terms**0.2
    return sum_log_terms - prod_pow

# ==============================================================================
# 2. DEFINICIÓN DE FUNCIONES DE RESTRICCIÓN Y LÍMITES (Idénticas)
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

def limites_dominio_f4_generacion() -> list[tuple[float, float]]:
    epsilon = 1e-6
    return [(2.0 + epsilon, 10.0 - epsilon)] * 10

def limites_dominio_f4_pdf() -> list[tuple[float, float]]:
    return [(-2.001, 10.0)] * 10

PROBLEMAS = {
    "f1": {"funcion_objetivo": f1, "funcion_restriccion": restriccion_f1, "limites_dominio_pdf": limites_dominio_f1, "limites_generacion": limites_dominio_f1, "dimension": 2},
    "f2": {"funcion_objetivo": f2, "funcion_restriccion": restriccion_f2, "limites_dominio_pdf": limites_dominio_f2, "limites_generacion": limites_dominio_f2, "dimension": 6},
    "f3": {"funcion_objetivo": f3, "funcion_restriccion": restriccion_f3, "limites_dominio_pdf": limites_dominio_f3, "limites_generacion": limites_dominio_f3, "dimension": 2},
    "f4": {"funcion_objetivo": f4, "funcion_restriccion": restriccion_f4, "limites_dominio_pdf": limites_dominio_f4_pdf, "limites_generacion": limites_dominio_f4_generacion, "dimension": 10},
}

# ==============================================================================
# 3. IMPLEMENTACIÓN DE PARTICLE SWARM OPTIMIZATION (PSO)
# ==============================================================================
class ParticleSwarmOptimization:
    def __init__(self, nombre_problema: str,
                 num_particulas: int, num_iteraciones: int,
                 w_inercia: float, c1_cognitivo: float, c2_social: float,
                 limite_velocidad_factor: float = 0.5): # Factor para Vmax = factor * rango_dominio
        """
        Inicializa el optimizador Particle Swarm Optimization.

        Args:
            nombre_problema (str): Clave del problema en PROBLEMAS.
            num_particulas (int): Número de partículas en el enjambre.
            num_iteraciones (int): Número máximo de iteraciones.
            w_inercia (float): Peso de inercia (w).
            c1_cognitivo (float): Coeficiente cognitivo (c1).
            c2_social (float): Coeficiente social (c2).
            limite_velocidad_factor (float): Factor para determinar la velocidad máxima
                                            permitida para una partícula (Vmax = factor * rango_variable).
        """
        if nombre_problema not in PROBLEMAS:
            raise ValueError(f"Problema '{nombre_problema}' no definido.")

        self.problema = PROBLEMAS[nombre_problema]
        self.funcion_objetivo = self.problema["funcion_objetivo"]
        self.funcion_restriccion = self.problema["funcion_restriccion"]
        self.limites_generacion = self.problema["limites_generacion"]()
        self.dimension = self.problema["dimension"]

        self.num_particulas = num_particulas
        self.num_iteraciones = num_iteraciones
        self.w_inercia = w_inercia
        self.c1_cognitivo = c1_cognitivo
        self.c2_social = c2_social

        # Calcular Vmax para cada dimensión
        self.v_max = np.array([(self.limites_generacion[d][1] - self.limites_generacion[d][0]) * limite_velocidad_factor
                               for d in range(self.dimension)])

        # Partículas
        self.posiciones = np.zeros((num_particulas, self.dimension))
        self.velocidades = np.zeros((num_particulas, self.dimension))
        self.fitness_particulas = np.full(num_particulas, float('inf'))

        # Mejores personales (pbest)
        self.pbest_posiciones = np.copy(self.posiciones)
        self.pbest_fitness = np.full(num_particulas, float('inf'))

        # Mejor global (gbest)
        self.gbest_posicion = np.zeros(self.dimension)
        self.gbest_fitness = float('inf')

        self.historial_convergencia = []

    def _inicializar_enjambre(self):
        for i in range(self.num_particulas):
            # Inicializar posición aleatoria válida
            while True:
                self.posiciones[i] = np.array([random.uniform(self.limites_generacion[d][0], self.limites_generacion[d][1])
                                               for d in range(self.dimension)])
                if self.funcion_restriccion(self.posiciones[i].tolist()):
                    break

            # Inicializar velocidad (e.g., a cero o pequeños valores aleatorios)
            # self.velocidades[i] = np.array([random.uniform(-self.v_max[d], self.v_max[d])
            # for d in range(self.dimension)])
            self.velocidades[i] = np.zeros(self.dimension)


            self.fitness_particulas[i] = self.funcion_objetivo(self.posiciones[i].tolist())

            # Inicializar pbest
            self.pbest_posiciones[i] = np.copy(self.posiciones[i])
            self.pbest_fitness[i] = self.fitness_particulas[i]

            # Actualizar gbest
            if self.pbest_fitness[i] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[i]
                self.gbest_posicion = np.copy(self.pbest_posiciones[i])

        self.historial_convergencia.append(self.gbest_fitness)

    def _aplicar_limites_posicion(self, posicion: np.ndarray) -> np.ndarray:
        for d in range(self.dimension):
            posicion[d] = max(self.limites_generacion[d][0], min(posicion[d], self.limites_generacion[d][1]))
        return posicion

    def _aplicar_limites_velocidad(self, velocidad: np.ndarray) -> np.ndarray:
        for d in range(self.dimension):
            velocidad[d] = max(-self.v_max[d], min(velocidad[d], self.v_max[d]))
        return velocidad

    def optimizar(self):
        print(f"Optimizando {self.problema['funcion_objetivo'].__name__} con Particle Swarm Optimization...")
        self._inicializar_enjambre()

        for iteracion in range(self.num_iteraciones):
            for i in range(self.num_particulas):
                # --- Actualizar velocidad ---
                r1 = random.random() # Componente aleatorio cognitivo
                r2 = random.random() # Componente aleatorio social

                termino_inercia = self.w_inercia * self.velocidades[i]
                termino_cognitivo = self.c1_cognitivo * r1 * (self.pbest_posiciones[i] - self.posiciones[i])
                termino_social = self.c2_social * r2 * (self.gbest_posicion - self.posiciones[i])

                self.velocidades[i] = termino_inercia + termino_cognitivo + termino_social
                self.velocidades[i] = self._aplicar_limites_velocidad(self.velocidades[i])

                # --- Actualizar posición ---
                self.posiciones[i] += self.velocidades[i]
                self.posiciones[i] = self._aplicar_limites_posicion(self.posiciones[i])

                # --- Evaluar fitness ---
                # Solo evaluar si la nueva posición es válida según las restricciones del PDF
                if self.funcion_restriccion(self.posiciones[i].tolist()):
                    self.fitness_particulas[i] = self.funcion_objetivo(self.posiciones[i].tolist())
                else:
                    self.fitness_particulas[i] = float('inf') # Penalizar

                # --- Actualizar pbest ---
                if self.fitness_particulas[i] < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = self.fitness_particulas[i]
                    self.pbest_posiciones[i] = np.copy(self.posiciones[i])

                    # --- Actualizar gbest ---
                    if self.pbest_fitness[i] < self.gbest_fitness:
                        self.gbest_fitness = self.pbest_fitness[i]
                        self.gbest_posicion = np.copy(self.pbest_posiciones[i])

            self.historial_convergencia.append(self.gbest_fitness)

            if (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}/{self.num_iteraciones} - Mejor Fitness Global: {self.gbest_fitness:.6e}")

        print("Optimización con Particle Swarm Optimization completada.")
        print(f"Mejor solución encontrada para {self.problema['funcion_objetivo'].__name__}:")
        print(f"  Variables: {[round(v, 5) for v in self.gbest_posicion]}")
        print(f"  Fitness: {self.gbest_fitness:.6e}")
        return self.gbest_posicion, self.historial_convergencia


# --- Parámetros de Particle Swarm Optimization (ejemplo) ---
# Estos son solo ejemplos, debes elegir y justificar 4 configuraciones
CONFIG_PSO_DEFAULT = {
    "num_particulas": 50,
    "num_iteraciones": 100,
    "w_inercia": 0.7,       # Típicamente [0.4, 0.9]
    "c1_cognitivo": 1.5,    # Típicamente [1.0, 2.0]
    "c2_social": 1.5,       # Típicamente [1.0, 2.0]
    "limite_velocidad_factor": 0.5 # Vmax = 0.5 * (rango_variable)
}

# ==============================================================================
# 4. EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":

    # --- Seleccionar el problema a optimizar ---
    nombre_del_problema_a_resolver = "f1" # Ejemplo con f1
    # nombre_del_problema_a_resolver = "f3"
    # nombre_del_problema_a_resolver = "f4"

    print(f"\n--- Resolviendo problema: {nombre_del_problema_a_resolver} con Particle Swarm Optimization ---")

    optimizador_pso = ParticleSwarmOptimization(
        nombre_problema=nombre_del_problema_a_resolver,
        num_particulas=CONFIG_PSO_DEFAULT["num_particulas"], # Corregido
        num_iteraciones=CONFIG_PSO_DEFAULT["num_iteraciones"], # Corregido
        w_inercia=CONFIG_PSO_DEFAULT["w_inercia"], # Corregido
        c1_cognitivo=CONFIG_PSO_DEFAULT["c1_cognitivo"], # Corregido
        c2_social=CONFIG_PSO_DEFAULT["c2_social"], # Corregido
        limite_velocidad_factor=CONFIG_PSO_DEFAULT["limite_velocidad_factor"] # Corregido
    )

    mejor_posicion, historial = optimizador_pso.optimizar()

    # (Cuando me pidas graficos_pso_optimizer.py, aquí pondremos la lógica de graficación)