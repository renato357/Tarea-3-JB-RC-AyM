import math
import random
import numpy as np
# Para operaciones numéricas, especialmente el muestreo gaussiano

# ==============================================================================
# 1. DEFINICIÓN DE FUNCIONES OBJETIVO
# ==============================================================================
# Fuente: Tarea 3 CIT3352, Funciones multimodales

def f1(variables: list[float]) -> float:
    """
    Función objetivo f1(x) = 4 - 4*x1^3 - 4*x1 + x2^2.
    Variables: [x1, x2]
    """
    if len(variables) != 2:
        raise ValueError("f1 espera 2 variables (x1, x2)")
    x1, x2 = variables[0], variables[1]
    return 4 - 4 * x1**3 - 4 * x1 + x2**2

def f2(variables: list[float]) -> float:
    """
    Función objetivo f2(x) = (1/899) * (sum_{i=1 to 6} (x_i^2 * 2^i) - 1745).
    Variables: [x1, x2, x3, x4, x5, x6]
    """
    if len(variables) != 6:
        raise ValueError("f2 espera 6 variables (x1 a x6)")

    sum_term = 0
    for i in range(6): # i va de 0 a 5
        # x_i en la fórmula es variables[i] (0-indexed)
        # 2^i en la fórmula es 2^(i+1) (si i es 1-indexed en fórmula)
        sum_term += (variables[i]**2) * (2**(i + 1))

    return (1/899) * (sum_term - 1745)

def f3(variables: list[float]) -> float:
    """
    Función objetivo f3(x) = (x1^6 + x2^4 - 17)^2 + (2*x1 + x2 - 4)^2.
    Variables: [x1, x2]
    """
    if len(variables) != 2:
        raise ValueError("f3 espera 2 variables (x1, x2)")
    x1, x2 = variables[0], variables[1]
    term1 = x1**6 + x2**4 - 17
    term2 = 2 * x1 + x2 - 4
    return term1**2 + term2**2

def f4(variables: list[float]) -> float:
    """
    Función objetivo f4(x) = sum_{i=1 to 10} [(ln(xi-2))^2 + (ln(10-xi))^2] - (prod_{i=1 to 10} xi)^0.2.
    Variables: [x1, ..., x10]
    Dominio estricto para evaluabilidad: 2 < xi < 10.
    """
    if len(variables) != 10:
        raise ValueError("f4 espera 10 variables (x1 a x10)")

    sum_log_terms = 0
    product_terms = 1.0

    for i in range(10):
        xi = variables[i]
        # Comprobación de dominio estricto para evaluabilidad de ln
        if not (2 < xi < 10):
            return float('inf') # Penalización alta si está fuera del dominio del logaritmo

        sum_log_terms += (math.log(xi - 2))**2 + (math.log(10 - xi))**2
        product_terms *= xi

    # (prod xi)^0.2 puede ser problemático si prod es negativo.
    # Dado que 2 < xi < 10, product_terms será positivo.
    if product_terms < 0: # Salvaguarda, aunque no debería ocurrir con 2 < xi < 10
        # Manejar raíz de número negativo (e.g., devolver inf o usar cmath)
        # Para este problema, xi > 2, así que el producto es positivo.
        # Si aún así fuera negativo, indicaría un problema.
        # Si product_terms es 0 y el exponente negativo, también sería inf.
         # prod xi^(1/5)
        prod_pow = - ((-product_terms)**0.2) if product_terms < 0 else product_terms**0.2

    else:
        prod_pow = product_terms**0.2

    return sum_log_terms - prod_pow

# ==============================================================================
# 2. DEFINICIÓN DE FUNCIONES DE RESTRICCIÓN Y LÍMITES DE DOMINIO
# ==============================================================================
# Estas funciones verifican las restricciones dadas en el PDF.

# --- Restricciones para f1 ---
def restriccion_f1(variables: list[float]) -> bool:
    x1, x2 = variables[0], variables[1]
    if not (-5 <= x1 <= 5): return False
    if not (-5 <= x2 <= 5): return False
    return True

def limites_dominio_f1() -> list[tuple[float, float]]:
    return [(-5.0, 5.0), (-5.0, 5.0)]

# --- Restricciones para f2 ---
def _verificar_dominio_valor_f2(xi: float) -> bool:
    return 0 <= xi <= 1

def restriccion_f2(variables: list[float]) -> bool:
    if len(variables) != 6:
        # print("Advertencia: restriccion_f2 esperaba 6 valores.") # Opcional
        return False
    for xi in variables:
        if not _verificar_dominio_valor_f2(xi):
            return False
    return True

def limites_dominio_f2() -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * 6

# --- Restricciones para f3 ---
def restriccion_f3(variables: list[float]) -> bool:
    x1, x2 = variables[0], variables[1]
    if not (-500 <= x1 <= 500): return False
    if not (-500 <= x2 <= 500): return False
    return True

def limites_dominio_f3() -> list[tuple[float, float]]:
    return [(-500.0, 500.0), (-500.0, 500.0)]

# --- Restricciones para f4 ---
# Nota: f4 tiene un dominio implícito más estricto (2 < xi < 10) para la evaluabilidad de ln.
# La función de restricción aquí verifica la restricción del PDF: -2.001 <= xi <= 10.
# El optimizador debería generar preferentemente en (2, 10) para f4.
def _verificar_dominio_valor_f4(xi: float) -> bool:
    # Restricción del PDF
    return -2.001 <= xi <= 10

def restriccion_f4(variables: list[float]) -> bool:
    if len(variables) != 10:
        # print("Advertencia: restriccion_f4 esperaba 10 valores.") # Opcional
        return False
    for xi in variables:
        if not _verificar_dominio_valor_f4(xi):
            return False
    return True

def limites_dominio_f4() -> list[tuple[float, float]]:
    # Para generación/clipping en ACO, es mejor usar el dominio evaluable de f4.
    # Ligeramente ajustado para evitar problemas en los bordes con log.
    epsilon = 1e-6
    return [(2.0 + epsilon, 10.0 - epsilon)] * 10

# Diccionario para acceder fácilmente a las funciones y sus propiedades
PROBLEMAS = {
    "f1": {"funcion_objetivo": f1, "funcion_restriccion": restriccion_f1, "limites_dominio": limites_dominio_f1, "dimension": 2},
    "f2": {"funcion_objetivo": f2, "funcion_restriccion": restriccion_f2, "limites_dominio": limites_dominio_f2, "dimension": 6},
    "f3": {"funcion_objetivo": f3, "funcion_restriccion": restriccion_f3, "limites_dominio": limites_dominio_f3, "dimension": 2},
    "f4": {"funcion_objetivo": f4, "funcion_restriccion": restriccion_f4, "limites_dominio": limites_dominio_f4, "dimension": 10},
}

# ==============================================================================
# 3. IMPLEMENTACIÓN DE ANT COLONY OPTIMIZATION (ACO) PARA DOMINIOS CONTINUOS
# ==============================================================================

class SolucionACO:
    def __init__(self, variables: list[float], valor_objetivo: float):
        self.variables = variables
        self.valor_objetivo = valor_objetivo

    def __lt__(self, other): # Para poder ordenar las soluciones (minimización)
        return self.valor_objetivo < other.valor_objetivo

class ACOContinuo:
    def __init__(self, nombre_problema: str,
                 num_hormigas: int, num_iteraciones: int,
                 tamano_archivo: int, q_param_seleccion: float, xi_param_desviacion: float):
        """
        Inicializa el optimizador ACO para dominios continuos.

        Args:
            nombre_problema (str): Clave del problema en el diccionario PROBLEMAS (e.g., "f1").
            num_hormigas (int): Número de hormigas (soluciones a generar por iteración).
            num_iteraciones (int): Número máximo de iteraciones.
            tamano_archivo (int): Número de mejores soluciones a guardar en el archivo (k).
            q_param_seleccion (float): Parámetro 'q' para la selección de soluciones del archivo.
                                      Controla la agudeza de la distribución de pesos.
                                      Valores pequeños dan más peso a las mejores soluciones (explotación).
            xi_param_desviacion (float): Parámetro 'xi' (o zeta) para el cálculo de la desviación estándar
                                         en el muestreo gaussiano. Controla la convergencia/diversidad.
                                         Valores más grandes -> mayor exploración.
        """
        if nombre_problema not in PROBLEMAS:
            raise ValueError(f"Problema '{nombre_problema}' no definido.")

        self.problema = PROBLEMAS[nombre_problema]
        self.funcion_objetivo = self.problema["funcion_objetivo"]
        self.funcion_restriccion = self.problema["funcion_restriccion"]
        self.limites_dominio_originales = self.problema["limites_dominio"]() # Limites del PDF
        self.dimension = self.problema["dimension"]

        # Para f4, usamos límites más estrictos para la generación interna en ACO
        if nombre_problema == "f4":
             self.limites_generacion = limites_dominio_f4() # [(2+eps, 10-eps)]*10
        else:
             self.limites_generacion = self.limites_dominio_originales


        self.num_hormigas = num_hormigas
        self.num_iteraciones = num_iteraciones
        self.tamano_archivo = tamano_archivo # k
        self.q_param_seleccion = q_param_seleccion
        self.xi_param_desviacion = xi_param_desviacion

        self.archivo_soluciones: list[SolucionACO] = []
        self.mejor_solucion_global: SolucionACO = None

        self.historial_convergencia = [] # Para guardar el mejor valor en cada iteración


    def _generar_solucion_aleatoria_valida(self) -> list[float]:
        """Genera una solución aleatoria dentro de los límites y que cumpla restricciones."""
        while True:
            variables = [random.uniform(self.limites_generacion[d][0], self.limites_generacion[d][1]) for d in range(self.dimension)]
            if self.funcion_restriccion(variables): # Chequea contra las restricciones del PDF
                 # Adicionalmente, para f4, la propia función objetivo ya penaliza si xi no está en (2,10)
                return variables

    def _evaluar_y_crear_solucion(self, variables: list[float]) -> SolucionACO:
        """Evalúa las variables y devuelve un objeto SolucionACO."""
        # Asegurarse de que las variables están dentro de los límites de generación/clipping
        variables_clipeadas = self._clipear_variables(variables)

        if not self.funcion_restriccion(variables_clipeadas): #
            # Si después de clipear aún no cumple la restricción general del PDF, penalizar.
            # Esto es una doble verificación, ya que la generación debería cuidar esto.
            valor_obj = float('inf')
        else:
            valor_obj = self.funcion_objetivo(variables_clipeadas)

        return SolucionACO(variables_clipeadas, valor_obj)

    def _clipear_variables(self, variables: list[float]) -> list[float]:
        """Asegura que las variables estén dentro de los límites de generación."""
        clipeadas = []
        for i in range(self.dimension):
            min_val, max_val = self.limites_generacion[i]
            val = max(min_val, min(variables[i], max_val))
            clipeadas.append(val)
        return clipeadas

    def _inicializar_archivo(self):
        """Llena el archivo con soluciones aleatorias válidas y las ordena."""
        self.archivo_soluciones = []
        for _ in range(self.tamano_archivo):
            vars_aleatorias = self._generar_solucion_aleatoria_valida()
            solucion = self._evaluar_y_crear_solucion(vars_aleatorias)
            self.archivo_soluciones.append(solucion)

        self.archivo_soluciones.sort() # Ordena por valor_objetivo (ascendente)
        self.mejor_solucion_global = self.archivo_soluciones[0]
        self.historial_convergencia.append(self.mejor_solucion_global.valor_objetivo)


    def _calcular_pesos_seleccion(self) -> list[float]:
        """Calcula los pesos omega_l para seleccionar soluciones del archivo."""
        pesos = []
        for l_rank in range(1, self.tamano_archivo + 1): # rank l de 1 a k
            numerador = math.exp(-((l_rank - 1)**2) / (2 * self.q_param_seleccion**2 * self.tamano_archivo**2))
            denominador = self.q_param_seleccion * self.tamano_archivo * math.sqrt(2 * math.pi)
            pesos.append(numerador / denominador)

        suma_pesos = sum(pesos)
        if suma_pesos == 0: # Evitar división por cero si todos los pesos son minúsculos
             return [1.0 / self.tamano_archivo] * self.tamano_archivo
        probabilidades = [p / suma_pesos for p in pesos]
        return probabilidades

    def _seleccionar_solucion_del_archivo(self, probabilidades_seleccion: list[float]) -> SolucionACO:
        """Selecciona una solución del archivo basada en las probabilidades."""
        return random.choices(self.archivo_soluciones, weights=probabilidades_seleccion, k=1)[0]

    def _generar_nueva_solucion_gaussiana(self, solucion_base: SolucionACO) -> list[float]:
        """
        Genera una nueva solución muestreando alrededor de la solucion_base.
        Para cada dimensión j, la nueva variable x_j se muestrea de N(solucion_base_j, sigma_j).
        sigma_j = xi_param * sum_e(|s_e_j - s_base_j|) / (k-1).
        """
        nuevas_variables = [0.0] * self.dimension
        for j_dim in range(self.dimension):
            suma_diferencias_abs = 0
            if self.tamano_archivo > 1:
                for s_e in self.archivo_soluciones: # s_e es una SolucionACO
                    suma_diferencias_abs += abs(s_e.variables[j_dim] - solucion_base.variables[j_dim])
                desviacion_std_j = self.xi_param_desviacion * (suma_diferencias_abs / (self.tamano_archivo - 1))
            else: # Si el archivo tiene solo una solución, usar una pequeña desviación para explorar
                desviacion_std_j = self.xi_param_desviacion * abs(self.limites_generacion[j_dim][1] - self.limites_generacion[j_dim][0]) * 0.1 # 10% del rango

            # Evitar desviación estándar cero o muy pequeña
            if desviacion_std_j < 1e-6:
                desviacion_std_j = 1e-6

            nuevas_variables[j_dim] = np.random.normal(loc=solucion_base.variables[j_dim], scale=desviacion_std_j)

        return self._clipear_variables(nuevas_variables)


    def optimizar(self):
        """Ejecuta el algoritmo ACO para encontrar el mínimo de la función."""
        print(f"Optimizando {self.problema['funcion_objetivo'].__name__} con ACO...")
        self._inicializar_archivo()

        probabilidades_seleccion = self._calcular_pesos_seleccion()

        for iteracion in range(self.num_iteraciones):
            nuevas_soluciones_hormigas = []
            for _ in range(self.num_hormigas):
                solucion_guia = self._seleccionar_solucion_del_archivo(probabilidades_seleccion)
                variables_generadas = self._generar_nueva_solucion_gaussiana(solucion_guia)

                # Evaluar y crear objeto SolucionACO
                # La función _evaluar_y_crear_solucion ya clipea y verifica restricciones del PDF
                # La función objetivo (ej. f4) maneja su propio dominio estricto (ej. 2 < xi < 10)
                nueva_solucion = self._evaluar_y_crear_solucion(variables_generadas)
                nuevas_soluciones_hormigas.append(nueva_solucion)

            # Actualizar el archivo con las nuevas soluciones
            self.archivo_soluciones.extend(nuevas_soluciones_hormigas)
            self.archivo_soluciones.sort() # Ordenar de mejor a peor
            self.archivo_soluciones = self.archivo_soluciones[:self.tamano_archivo] # Mantener solo las k mejores

            # Actualizar la mejor solución global encontrada
            if self.archivo_soluciones[0] < self.mejor_solucion_global:
                self.mejor_solucion_global = self.archivo_soluciones[0]

            self.historial_convergencia.append(self.mejor_solucion_global.valor_objetivo)

            if (iteracion + 1) % 10 == 0: # Imprimir progreso cada 10 iteraciones
                print(f"Iteración {iteracion + 1}/{self.num_iteraciones} - Mejor Valor: {self.mejor_solucion_global.valor_objetivo:.6e}")

        print("Optimización completada.")
        print(f"Mejor solución encontrada para {self.problema['funcion_objetivo'].__name__}:")
        print(f"  Variables: {[round(v, 5) for v in self.mejor_solucion_global.variables]}")
        print(f"  Valor Objetivo: {self.mejor_solucion_global.valor_objetivo:.6e}")
        return self.mejor_solucion_global, self.historial_convergencia


# --- Parámetros del ACO (ejemplo, ajustar para cada función y experimento) ---
# Estos son solo ejemplos, debes elegir y justificar 4 configuraciones
CONFIG_ACO_DEFAULT = {
    "num_hormigas": 30,         # Número de hormigas
    "num_iteraciones": 100,     # Número de iteraciones
    "tamano_archivo": 10,       # k: tamaño del archivo de soluciones (e.g., 10-20% de num_hormigas)
    "q_param_seleccion": 0.2,   # q: para la selección de soluciones (e.g., 0.1 - 0.5)
    "xi_param_desviacion": 0.85 # xi: para el cálculo de sigma (e.g., 0.5 - 1.5)
}
# ==============================================================================
# 4. EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":

    # --- Seleccionar el problema a optimizar ---
    # Puedes cambiar "f1" por "f2", "f3", o "f4"
    nombre_del_problema_a_resolver = "f1" # Ejemplo con f1
    # nombre_del_problema_a_resolver = "f4" # Prueba f4

    print(f"\n--- Resolviendo problema: {nombre_del_problema_a_resolver} ---")

    # Crear instancia del optimizador ACO
    optimizador_aco = ACOContinuo(
        nombre_problema=nombre_del_problema_a_resolver,
        num_hormigas=CONFIG_ACO_DEFAULT["num_hormigas"], # Corregido para usar CONFIG_ACO_DEFAULT
        num_iteraciones=CONFIG_ACO_DEFAULT["num_iteraciones"], # Corregido
        tamano_archivo=CONFIG_ACO_DEFAULT["tamano_archivo"], # Corregido
        q_param_seleccion=CONFIG_ACO_DEFAULT["q_param_seleccion"], # Corregido
        xi_param_desviacion=CONFIG_ACO_DEFAULT["xi_param_desviacion"] # Corregido
    )

    # Ejecutar la optimización
    mejor_solucion, historial = optimizador_aco.optimizar()

    # (Aquí es donde harías más ejecuciones, guardarías resultados, graficarías, etc.)
    # Ejemplo de cómo podrías graficar la convergencia (necesitas matplotlib):
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(historial)
    # plt.title(f"Convergencia de ACO para {nombre_del_problema_a_resolver}")
    # plt.xlabel("Iteración")
    # plt.ylabel("Mejor Valor Objetivo")
    # plt.grid(True)
    # plt.show()

    # Para tu tarea, recuerda:
    # 1. Realizar 10 ejecuciones por cada función y por cada configuración de parámetros.
    # 2. Usar al menos 4 configuraciones de parámetros distintas, justificándolas.
    # 3. Mostrar resultados, incluyendo gráficos de convergencia.
    # 4. Analizar y concluir sobre los resultados.