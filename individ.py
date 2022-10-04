# класс особи хранящий в себе:
# геном genom
# модель нейросети model
# приспособленность особи fitness
class Individ(object):
    def __init__(self, genom, model):
        self.genom = genom
        self.model = model
        self.fitness = 0

    def update_fitness(self, fitness_function, environment):
        self.fitness = fitness_function(self.model, environment)