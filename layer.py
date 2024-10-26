import numpy as np

from neuron import Neuron


class Layer:
    """
    Una capa de neuronas en la red neuronal.

    Atributos:
        neurons (list): Lista de objetos Neuron en la capa.
    """

    def __init__(self, num_neurons, input_size):
        """
        Inicializa una capa con un número especificado de neuronas.

        Parámetros:
            num_neurons (int): Número de neuronas en la capa.
            input_size (int): Número de entradas para cada neurona en la capa.
        """
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        """
        Calcula el paso hacia adelante para la capa.

        Parámetros:
            inputs (ndarray): Los valores de entrada a la capa.

        Regresa:
            ndarray: Las salidas activadas de las neuronas en la capa.
        """
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return outputs

    def backward(self, d_outputs, learning_rate):
        """
        Calcula el paso hacia atrás y actualiza las neuronas en la capa.

        Parámetros:
            d_outputs (ndarray): Los gradientes de la pérdida con respecto a las salidas de la capa.
            learning_rate (float): La tasa de aprendizaje para las actualizaciones de peso.

        Regresa:
            ndarray: Los gradientes de la pérdida con respecto a las entradas a la capa.
        """
        d_inputs = np.zeros(len(self.neurons[0].input))  # Inicializar el gradiente respecto a las entradas
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una capa con 3 neuronas, cada una recibiendo 4 entradas
    layer = Layer(3, 4)

    # Entradas de ejemplo (4 características)
    inputs = np.array([1.0, 0.5, -1.5, 2.0])

    # Calcular la salida de la capa
    layer_output = layer.forward(inputs)

    print("Salida de la capa:", layer_output)
