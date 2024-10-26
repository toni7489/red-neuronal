import numpy as np
from layer import Layer


class NeuralNetwork:
    """
    Una red neuronal simple de alimentación hacia adelante.

    Atributos:
        layers (list): Lista de objetos Layer en la red neuronal.
    """

    def __init__(self):
        """
        Inicializa una red neuronal vacía sin capas.
        """
        self.layers = []
        self.loss_list = []

    def add_layer(self, num_neurons, input_size):
        """
        Agrega una capa a la red neuronal.

        Parámetros:
            num_neurons (int): El número de neuronas en la nueva capa.
            input_size (int): El número de entradas a la nueva capa (o neuronas).
        """
        if not self.layers:
            self.layers.append(Layer(num_neurons, input_size))
        else:
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size))

    def forward(self, inputs):
        """
        Calcula el paso hacia adelante a través de toda la red.

        Parámetros:
            inputs (ndarray): Los valores de entrada a la red.

        Regresa:
            ndarray: La salida de la red después del paso hacia adelante.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, learning_rate):
        """
        Calcula el paso hacia atrás a través de toda la red.

        Parámetros:
            loss_gradient (ndarray): El gradiente de la pérdida con respecto a la salida de la red.
            learning_rate (float): La tasa de aprendizaje para las actualizaciones de peso.
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Entrena la red neuronal utilizando los datos de entrenamiento dados.

        Parámetros:
            X (ndarray): Los datos de entrenamiento de entrada (características).
            y (ndarray): Los valores de salida objetivo (etiquetas).
            epochs (int): El número de iteraciones de entrenamiento.
            learning_rate (float): El tamaño del paso para las actualizaciones de peso.
        """
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                # Paso hacia adelante
                output = self.forward(X[i])
                # Calcular la pérdida (error cuadrático medio)
                loss += np.mean((y[i] - output) ** 2)
                # Paso hacia atrás (gradiente de la pérdida con respecto a la salida)
                loss_gradient = 2 * (output - y[i])
                self.backward(loss_gradient, learning_rate)
            loss /= len(X)
            self.loss_list.append(loss)
            if epoch % 100 == 0:
                print(f"Época {epoch}, Pérdida: {loss}")

    def predict(self, X):
        """
        Genera predicciones para los datos de entrada.

        Parámetros:
            X (ndarray): Los datos de entrada para la predicción.

        Regresa:
            ndarray: Las salidas predichas de la red.
        """
        predictions = []
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        return np.array(predictions)


if __name__ == "__main__":
    # Ejemplo de uso:
    # Crear una red neuronal con 3 características de entrada, 1 capa oculta y 1 capa de salida
    nn = NeuralNetwork()

    # Agregar capas: el tamaño de entrada para la primera capa oculta es 3 (por ejemplo, 3 características de entrada)
    nn.add_layer(num_neurons=3, input_size=3)  # Primera capa oculta con 3 neuronas
    nn.add_layer(num_neurons=3, input_size=3)  # Segunda capa oculta con 3 neuronas
    nn.add_layer(num_neurons=1, input_size=3)  # Capa de salida con 1 neurona

    # Datos de entrenamiento ficticios (X: entrada, y: objetivo)
    X = np.array([[0.5, 0.2, 0.1],
                  [0.9, 0.7, 0.3],
                  [0.4, 0.5, 0.8]])
    
    y = np.array([[0.3, 0.6, 0.9]]).T

    # Entrenar la red
    nn.train(X, y, epochs=100000, learning_rate=0.1)

    # Predecir utilizando la red entrenada
    predictions = nn.predict(X)
    print(f"Predicciones: {predictions}")
