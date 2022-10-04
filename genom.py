import random
import numpy as np


# класс генома нейросети
# neurons - массив целых чисел обозначающих количество нейронов в скрытых слоях нейросети
# layers - количество слоев в нейросети
class Genom(object):
    def __init__(self, neurons):
        self.neurons = neurons
        self.layers = len(self.neurons)

    # генерирует случайный геном с количеством скрытых слоев от 0 до 5 и количеством нейронов в слое от 1 до 50
    def random_genom(self):
        self.layers = random.randint(1, 5)
        self.neurons = np.random.randint(1, 50, self.layers)

    def mutate(self, probability):
        for i in range(self.layers):
            if random.random() < probability:
                self.neurons[i] = random.randint(0, 50)
        if random.random() < probability / 10:
            self.neurons = np.append(self.neurons, random.randint(1, 50))
        self.neurons = np.delete(self.neurons, np.where(self.neurons == 0))
        self.layers = len(self.neurons)