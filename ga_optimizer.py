# ga_optimizer.py

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

def _verificar_dominio_valor_f4(xi: float) -> bool: return -2.001 <= xi <= 10 # Restricción del PDF
def restriccion_f4(variables: list[float]) -> bool:
    if len(variables) != 10: return False
    return all(_verificar_dominio_valor_f4(xi) for xi in variables)

def limites_dominio_f4_generacion() -> list[tuple[float, float]]: # Para generación en GA para f4
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
# 3. IMPLEMENTACIÓN DE GENETIC ALGORITHM (GA)
# ==============================================================================
class IndividuoGA:
    def __init__(self, variables: np.ndarray, fitness: float = float('inf')):
        self.variables = variables # Cromosoma
        self.fitness = fitness

    def __lt__(self, other): # Para poder ordenar por fitness (minimización)
        return self.fitness < other.fitness

class GeneticAlgorithm:
    def __init__(self, nombre_problema: str,
                 tamano_poblacion: int, prob_crossover: float,
                 prob_mutacion_gen: float, num_generaciones: int,
                 tamano_torneo: int = 3,
                 sigma_mutacion: float = 0.1, # Desviación estándar para mutación gaussiana
                 tasa_elitismo: float = 0.1): # Porcentaje de élites
        """
        Inicializa el Algoritmo Genético.

        Args:
            nombre_problema (str): Clave del problema en PROBLEMAS.
            tamano_poblacion (int): Número de individuos en la población.
            prob_crossover (float): Probabilidad de que se realice cruce entre dos padres.
            prob_mutacion_gen (float): Probabilidad de que cada gen de un individuo mute.
            num_generaciones (int): Número máximo de generaciones.
            tamano_torneo (int): Número de individuos que participan en cada torneo de selección.
            sigma_mutacion (float): Desviación estándar para la mutación gaussiana.
            tasa_elitismo (float): Proporción de los mejores individuos que pasan directamente
                                   a la siguiente generación.
        """
        if nombre_problema not in PROBLEMAS:
            raise ValueError(f"Problema '{nombre_problema}' no definido.")

        self.problema = PROBLEMAS[nombre_problema]
        self.funcion_objetivo = self.problema["funcion_objetivo"]
        self.funcion_restriccion = self.problema["funcion_restriccion"]
        self.limites_generacion = self.problema["limites_generacion"]()
        self.dimension = self.problema["dimension"]

        self.tamano_poblacion = tamano_poblacion
        self.prob_crossover = prob_crossover
        self.prob_mutacion_gen = prob_mutacion_gen
        self.num_generaciones = num_generaciones
        self.tamano_torneo = tamano_torneo
        self.sigma_mutacion = sigma_mutacion # sigma = rango_variable * sigma_mutacion_relativa
        self.num_elites = int(tamano_poblacion * tasa_elitismo)

        self.poblacion: list[IndividuoGA] = []
        self.mejor_individuo_global: IndividuoGA = None
        self.historial_convergencia = []

    def _crear_individuo_aleatorio(self) -> IndividuoGA:
        while True:
            variables = np.array([random.uniform(self.limites_generacion[d][0], self.limites_generacion[d][1])
                                  for d in range(self.dimension)])
            if self.funcion_restriccion(variables.tolist()):
                fitness = self.funcion_objetivo(variables.tolist())
                return IndividuoGA(variables, fitness)

    def _inicializar_poblacion(self):
        self.poblacion = [self._crear_individuo_aleatorio() for _ in range(self.tamano_poblacion)]
        self.poblacion.sort() # Ordenar por fitness
        self.mejor_individuo_global = self.poblacion[0]
        self.historial_convergencia.append(self.mejor_individuo_global.fitness)

    def _evaluar_individuo(self, individuo: IndividuoGA):
        # Aplicar límites primero
        individuo.variables = self._aplicar_limites_individuo(individuo.variables)
        if self.funcion_restriccion(individuo.variables.tolist()):
            individuo.fitness = self.funcion_objetivo(individuo.variables.tolist())
        else:
            individuo.fitness = float('inf') # Penalización alta

    def _aplicar_limites_individuo(self, variables: np.ndarray) -> np.ndarray:
        for d in range(self.dimension):
            variables[d] = max(self.limites_generacion[d][0], min(variables[d], self.limites_generacion[d][1]))
        return variables

    def _seleccion_torneo(self) -> IndividuoGA:
        participantes = random.sample(self.poblacion, self.tamano_torneo)
        participantes.sort() # El mejor (menor fitness) estará primero
        return participantes[0]

    def _crossover_aritmetico_simple(self, padre1: IndividuoGA, padre2: IndividuoGA) -> tuple[IndividuoGA, IndividuoGA]:
        vars1, vars2 = np.copy(padre1.variables), np.copy(padre2.variables)
        if random.random() < self.prob_crossover:
            alpha = random.random() # Factor de mezcla
            hijo1_vars = alpha * padre1.variables + (1 - alpha) * padre2.variables
            hijo2_vars = (1 - alpha) * padre1.variables + alpha * padre2.variables

            hijo1 = IndividuoGA(self._aplicar_limites_individuo(hijo1_vars))
            hijo2 = IndividuoGA(self._aplicar_limites_individuo(hijo2_vars))
            return hijo1, hijo2
        else:
            # Si no hay crossover, los hijos son clones de los padres
            return IndividuoGA(vars1), IndividuoGA(vars2)


    def _mutacion_gaussiana(self, individuo: IndividuoGA):
        for i in range(self.dimension):
            if random.random() < self.prob_mutacion_gen:
                # La desviación estándar de la mutación puede ser relativa al rango de la variable
                rango_variable = self.limites_generacion[i][1] - self.limites_generacion[i][0]
                sigma_efectiva = rango_variable * self.sigma_mutacion # O un valor absoluto pequeño
                if sigma_efectiva < 1e-6 : sigma_efectiva = 1e-6 # Evitar sigma cero

                individuo.variables[i] += random.gauss(0, sigma_efectiva)
        individuo.variables = self._aplicar_limites_individuo(individuo.variables)


    def optimizar(self):
        print(f"Optimizando {self.problema['funcion_objetivo'].__name__} con Genetic Algorithm...")
        self._inicializar_poblacion()

        for generacion in range(self.num_generaciones):
            nueva_poblacion = []

            # Elitismo: los mejores individuos pasan directamente
            if self.num_elites > 0:
                elites = self.poblacion[:self.num_elites]
                nueva_poblacion.extend([IndividuoGA(np.copy(elite.variables), elite.fitness) for elite in elites])

            # Generar el resto de la población mediante selección, crossover y mutación
            num_descendientes_necesarios = self.tamano_poblacion - self.num_elites

            descendientes_generados = 0
            while descendientes_generados < num_descendientes_necesarios :
                padre1 = self._seleccion_torneo()
                padre2 = self._seleccion_torneo()

                hijo1, hijo2 = self._crossover_aritmetico_simple(padre1, padre2)

                self._mutacion_gaussiana(hijo1)
                self._mutacion_gaussiana(hijo2)

                self._evaluar_individuo(hijo1) # Evaluar después de mutación y asegurar límites
                nueva_poblacion.append(hijo1)
                descendientes_generados +=1

                if descendientes_generados < num_descendientes_necesarios:
                    self._evaluar_individuo(hijo2)
                    nueva_poblacion.append(hijo2)
                    descendientes_generados +=1

            self.poblacion = nueva_poblacion[:self.tamano_poblacion] # Asegurar tamaño de población
            self.poblacion.sort() # Ordenar por fitness para la siguiente generación y elitismo

            if self.poblacion[0].fitness < self.mejor_individuo_global.fitness:
                self.mejor_individuo_global = IndividuoGA(np.copy(self.poblacion[0].variables), self.poblacion[0].fitness)

            self.historial_convergencia.append(self.mejor_individuo_global.fitness)

            if (generacion + 1) % 10 == 0:
                print(f"Generación {generacion + 1}/{self.num_generaciones} - Mejor Fitness: {self.mejor_individuo_global.fitness:.6e}")

        print("Optimización con Genetic Algorithm completada.")
        print(f"Mejor solución encontrada para {self.problema['funcion_objetivo'].__name__}:")
        print(f"  Variables: {[round(v, 5) for v in self.mejor_individuo_global.variables]}")
        print(f"  Fitness: {self.mejor_individuo_global.fitness:.6e}")
        return self.mejor_individuo_global, self.historial_convergencia


# --- Parámetros del Genetic Algorithm (ejemplo) ---
CONFIG_GA_DEFAULT = {
    "tamano_poblacion": 100,
    "prob_crossover": 0.8,
    "prob_mutacion_gen": 0.1, # Probabilidad de mutar cada gen individualmente
    "num_generaciones": 100,
    "tamano_torneo": 3,
    "sigma_mutacion": 0.1, # Factor para la desviación de la mutación gaussiana (relativo al rango)
    "tasa_elitismo": 0.05   # 5% de los mejores individuos pasan directamente
}

# ==============================================================================
# 4. EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":

    # --- Seleccionar el problema a optimizar ---
    nombre_del_problema_a_resolver = "f1" # Ejemplo con f1
    # nombre_del_problema_a_resolver = "f2"
    # nombre_del_problema_a_resolver = "f4"

    print(f"\n--- Resolviendo problema: {nombre_del_problema_a_resolver} con Genetic Algorithm ---")

    optimizador_ga = GeneticAlgorithm(
        nombre_problema=nombre_del_problema_a_resolver,
        tamano_poblacion=CONFIG_GA_DEFAULT["tamano_poblacion"], # Corregido
        prob_crossover=CONFIG_GA_DEFAULT["prob_crossover"], # Corregido
        prob_mutacion_gen=CONFIG_GA_DEFAULT["prob_mutacion_gen"], # Corregido
        num_generaciones=CONFIG_GA_DEFAULT["num_generaciones"], # Corregido
        tamano_torneo=CONFIG_GA_DEFAULT["tamano_torneo"], # Corregido
        sigma_mutacion=CONFIG_GA_DEFAULT["sigma_mutacion"], # Corregido
        tasa_elitismo=CONFIG_GA_DEFAULT["tasa_elitismo"] # Corregido
    )

    mejor_individuo, historial = optimizador_ga.optimizar()

    # (Cuando me pidas graficos_ga_optimizer.py, aquí pondremos la lógica de graficación)