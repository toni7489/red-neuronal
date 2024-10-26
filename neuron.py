import numpy as np


class Neuron:
    """
    Una sola neurona en la red neuronal.

    Atributos:
        weights (ndarray): Pesos asociados a las entradas.
        bias (float): Término de sesgo añadido a la suma ponderada de la neurona.
        output (float): Salida de la neurona después de la activación.
        input (ndarray): Entradas a la neurona durante el paso hacia adelante.
        dweights (ndarray): Gradiente de la pérdida con respecto a los pesos.
        dbias (float): Gradiente de la pérdida con respecto al sesgo.
    """

    def __init__(self, input_size):
        """
        Inicializa una neurona con pesos y sesgo aleatorios.

        Parámetros:
            input_size (int): El número de entradas a la neurona.
        """
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.output = 0
        self.input = None
        self.dweights = np.zeros_like(self.weights)
        self.dbias = 0

    def activate(self, x):
        """
        Aplica la función de activación sigmoide.

        Parámetros:
            x (float): El valor de entrada.

        Regresa:
            float: Valor de salida activada.
        """
        return 1 / (1 + np.exp(-x))

    def activate_derivative(self, x):
        """
        Calcula la derivada de la función sigmoide.

        Parámetros:
            x (float): El valor de salida activada.

        Regresa:
            float: Derivada de la sigmoide.
        """
        return x * (1 - x)

    def forward(self, inputs):
        """
        Calcula el paso hacia adelante para la neurona.

        Parámetros:
            inputs (ndarray): Los valores de entrada a la neurona.

        Regresa:
            float: La salida activada de la neurona.
        """
        self.input = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output

    def backward(self, d_output, learning_rate):
        """
        Calcula el paso hacia atrás y actualiza pesos y sesgo.

        Parámetros:
            d_output (float): El gradiente de la pérdida con respecto a la salida.
            learning_rate (float): La tasa de aprendizaje para las actualizaciones de peso.

        Regresa:
            ndarray: El gradiente de la pérdida con respecto a la entrada.
        """
        d_activation = d_output * self.activate_derivative(self.output)
        self.dweights = np.dot(self.input, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weights)
        # Actualizar pesos y sesgos
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
        return d_input


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una neurona con 3 entradas
    neuron = Neuron(3)

    # Entradas de ejemplo
    inputs = np.array([1.5, 2.0, -1.0])

    # Calcular la salida de la neurona
    output = neuron.forward(inputs)

    print("Salida de la neurona:", output)
