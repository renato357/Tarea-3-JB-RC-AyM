# de_optimizer.py

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
    # product_terms será positivo dado 2 < xi < 10
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

def _verificar_dominio_valor_f4(xi: float) -> bool: return -2.001 <= xi <= 10 # Restricción del PDF
def restriccion_f4(variables: list[float]) -> bool:
    if len(variables) != 10: return False
    return all(_verificar_dominio_valor_f4(xi) for xi in variables)

def limites_dominio_f4_generacion() -> list[tuple[float, float]]: # Para generación en DE para f4
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
# 3. IMPLEMENTACIÓN DE DIFFERENTIAL EVOLUTION (DE)
# ==============================================================================
class DifferentialEvolution:
    def __init__(self, nombre_problema: str,
                 tamano_poblacion: int, factor_mutacion_F: float,
                 prob_crossover_CR: float, num_generaciones: int,
                 estrategia_mutacion: str = "rand/1", semilla: int = None): # MODIFICADO: Añadir semilla
        """
        Inicializa el optimizador Differential Evolution.

        Args:
            nombre_problema (str): Clave del problema en PROBLEMAS.
            tamano_poblacion (int): Número de individuos en la población (NP).
            factor_mutacion_F (float): Factor de diferenciación (F), típicamente en [0.4, 1.0].
            prob_crossover_CR (float): Probabilidad de cruce (CR), típicamente en [0.0, 1.0].
            num_generaciones (int): Número máximo de generaciones (iteraciones).
            estrategia_mutacion (str): Estrategia de mutación, por defecto "rand/1".
            semilla (int, optional): Semilla para la generación de números aleatorios. Por defecto None.
        """
        if nombre_problema not in PROBLEMAS:
            raise ValueError(f"Problema '{nombre_problema}' no definido.")

        self.problema = PROBLEMAS[nombre_problema]
        self.funcion_objetivo = self.problema["funcion_objetivo"]
        self.funcion_restriccion = self.problema["funcion_restriccion"]
        self.limites_generacion = self.problema["limites_generacion"]()
        self.dimension = self.problema["dimension"]

        self.NP = tamano_poblacion
        self.F = factor_mutacion_F
        self.CR = prob_crossover_CR
        self.num_generaciones = num_generaciones
        self.estrategia_mutacion = estrategia_mutacion
        self.semilla = semilla # MODIFICADO: Guardar semilla

        self.poblacion = np.zeros((self.NP, self.dimension))
        self.fitness_poblacion = np.full(self.NP, float('inf'))

        self.mejor_individuo_global = np.zeros(self.dimension)
        self.mejor_fitness_global = float('inf')
        self.historial_convergencia = [] # Se inicializa aquí, se llena en optimizar

    def _inicializar_poblacion(self):
        # La semilla se establece en optimizar() antes de llamar a esta función.
        # Por lo tanto, todas las operaciones aleatorias aquí usarán esa semilla.
        for i in range(self.NP):
            # Generar individuo aleatorio válido
            while True:
                self.poblacion[i] = np.array([random.uniform(self.limites_generacion[d][0], self.limites_generacion[d][1])
                                              for d in range(self.dimension)])
                if self.funcion_restriccion(self.poblacion[i].tolist()):
                    break

            self.fitness_poblacion[i] = self.funcion_objetivo(self.poblacion[i].tolist())

            if self.fitness_poblacion[i] < self.mejor_fitness_global:
                self.mejor_fitness_global = self.fitness_poblacion[i]
                self.mejor_individuo_global = np.copy(self.poblacion[i])
        # El historial de convergencia se actualiza en optimizar() después de esta llamada.

    def _aplicar_limites_vector(self, vector: np.ndarray) -> np.ndarray:
        """Aplica los límites de generación a un vector."""
        for d in range(self.dimension):
            vector[d] = max(self.limites_generacion[d][0], min(vector[d], self.limites_generacion[d][1]))
        return vector

    def optimizar(self):
        # MODIFICADO: Configurar semilla si se proporcionó, al inicio de la optimización.
        if self.semilla is not None:
            random.seed(self.semilla)
            np.random.seed(self.semilla)

        # MODIFICADO: Reiniciar el estado para una ejecución limpia
        self.historial_convergencia = []
        self.mejor_fitness_global = float('inf')
        self.mejor_individuo_global = np.zeros(self.dimension) # Reiniciar también el mejor individuo
        self.poblacion = np.zeros((self.NP, self.dimension))
        self.fitness_poblacion = np.full(self.NP, float('inf'))


        # El print de `main_graficos_de.py` es más informativo sobre la ejecución general.
        # Este print es útil si se ejecuta `de_optimizer.py` directamente.
        # print(f"Optimizando {self.problema['funcion_objetivo'].__name__} con DE (Semilla: {self.semilla if self.semilla is not None else 'No especificada'})...")
        
        self._inicializar_poblacion() # La semilla ya está activa para esta llamada.
        
        # Añadir el primer mejor fitness al historial DESPUÉS de la inicialización.
        self.historial_convergencia.append(self.mejor_fitness_global)

        for generacion in range(self.num_generaciones):
            for i in range(self.NP):
                # --- 1. Mutación (DE/rand/1) ---
                idxs = [idx for idx in range(self.NP) if idx != i]
                if len(idxs) < 3 :
                    vector_mutante = np.copy(self.poblacion[i])
                else:
                    r1, r2, r3 = random.sample(idxs, 3)
                    vector_mutante = self.poblacion[r1] + self.F * (self.poblacion[r2] - self.poblacion[r3])
                    vector_mutante = self._aplicar_limites_vector(vector_mutante)

                # --- 2. Crossover (Binomial) ---
                vector_prueba = np.copy(self.poblacion[i])
                j_rand = random.randint(0, self.dimension - 1)

                for j in range(self.dimension):
                    if random.random() < self.CR or j == j_rand:
                        vector_prueba[j] = vector_mutante[j]

                # --- 3. Selección ---
                if self.funcion_restriccion(vector_prueba.tolist()):
                    fitness_prueba = self.funcion_objetivo(vector_prueba.tolist())
                else:
                    fitness_prueba = float('inf')

                if fitness_prueba < self.fitness_poblacion[i]:
                    self.poblacion[i] = vector_prueba
                    self.fitness_poblacion[i] = fitness_prueba

                    if fitness_prueba < self.mejor_fitness_global:
                        self.mejor_fitness_global = fitness_prueba
                        self.mejor_individuo_global = np.copy(vector_prueba)

            self.historial_convergencia.append(self.mejor_fitness_global)

            # Los prints de progreso detallado se omiten aquí para no saturar la consola
            # cuando se ejecuta desde main_graficos_de.py
            # if (generacion + 1) % 10 == 0:
            #     print(f"Generación {generacion + 1}/{self.num_generaciones} - Mejor Fitness: {self.mejor_fitness_global:.6e}")

        # Los prints de resultado final se omiten aquí por la misma razón.
        # print("Optimización con Differential Evolution completada.")
        # print(f"Mejor solución encontrada para {self.problema['funcion_objetivo'].__name__}:")
        # print(f"  Variables: {[round(v, 5) for v in self.mejor_individuo_global]}")
        # print(f"  Fitness: {self.mejor_fitness_global:.6e}")
        return self.mejor_individuo_global, self.historial_convergencia


# --- Parámetros de Differential Evolution (ejemplo) ---
CONFIG_DE_DEFAULT = {
    "tamano_poblacion": 50,
    "factor_mutacion_F": 0.7,
    "prob_crossover_CR": 0.8,
    "num_generaciones": 100
}

# ==============================================================================
# 4. EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":

    nombre_del_problema_a_resolver = "f1" 
    print(f"\n--- Resolviendo problema: {nombre_del_problema_a_resolver} con Differential Evolution (Ejemplo con semilla) ---")

    optimizador_de = DifferentialEvolution(
        nombre_problema=nombre_del_problema_a_resolver,
        semilla=42, # Ejemplo de uso de semilla
        **CONFIG_DE_DEFAULT
    )

    mejor_individuo, historial = optimizador_de.optimizar()
    
    if historial:
        print(f"Mejor solución para {nombre_del_problema_a_resolver} con semilla 42:")
        print(f"  Variables: {[round(v, 5) for v in mejor_individuo]}")
        print(f"  Fitness: {historial[-1]:.6e}")
    else:
        print(f"No se obtuvo historial para {nombre_del_problema_a_resolver} con semilla 42.")

