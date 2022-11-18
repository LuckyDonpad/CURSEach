import numpy as np
import random
import tournament
from generators import *

# принимает модель нейросети model, объект класса среды обитания environment.py
# возвращает приспособленность индивида к заданной среде
def get_model_fitness(model, environment):
    m_history = model.fit(verbose=1, x=environment.train_x, y=environment.train_y,
                          validation_data=(environment.val_x, environment.val_y),
                          batch_size=32, epochs=100)
    return max(m_history.history['val_accuracy']

# обновляет параметр приспособленности каждого индивида в популяции individ_pop
# высчитываемый при помощи функции fitness_function в среде environment.py
def update_pop_fitness(individ_pop, fitness_function, environment):
    for individ in individ_pop:
        if individ.fitness == 0:
            individ.update_fitness(fitness_function=fitness_function,
                                   environment=environment)


# функция скрещивания индивидов
# принимает два объекта класса Individ: ind_1 ind_2 и вероятность мутации probability
# возвращает объект класса Individ полученный путем скрещивания и мутации
def crossingover(ind_1, ind_2, probability):
    cross_point_1 = 0 if len(ind_1.genom.neurons) == 0 else random.randint(0, len(ind_1.genom.neurons))
    cross_point_2 = 0 if len(ind_2.genom.neurons) == 0 else random.randint(0, len(ind_2.genom.neurons))
    new_genom = Genom(
        np.concatenate((ind_1.genom.neurons[0:cross_point_1], ind_2.genom.neurons[cross_point_2: -1])))
    new_genom.mutate(probability)
    new_model = generate_neural_network_from_genom(new_genom, shapes.INPUT_SHAPE, shapes.OUT_CLASSES)
    new_individ = Individ(new_genom, new_model)
    return new_individ


# функция порождения новой популяции индивидов
# принимает:
# массив объектов класса Individ individ_pop
# функцию скрещивания и мутации crossingover
# вероятность мутации probability
# возвращает массив объектов класса Individ - родителей и их потомков

def reproduction(individ_pop, crossingover, probability):
    new_pop = []
    for i in range(len(individ_pop)):
        index_1 = random.randint(0, len(individ_pop) - 1)
        index_2 = random.randint(0, len(individ_pop) - 1)
        while index_1 == index_2:
            index_2 = random.randint(0, len(individ_pop) - 1)
        new_pop.append(crossingover(individ_pop[index_1], individ_pop[index_2], probability))
    for individ in individ_pop:
        new_pop.append(individ)
    return new_pop


# функция эволюции
# принимает:
# массив объектов класса Individ individ_pop
# функцию скрещивания и мутации crossingover
# вероятность мутации probability
# функцию преспособленности fitness_func
# объект Класса Environment
# возвращает массив объектов класса Individ - новое поколение индивидов

def evolve(individ_pop, fitness_func, environment, crossingover, probability):
    update_pop_fitness(individ_pop, fitness_func, environment)
    individ_pop = tournament.tournament_selection(individ_pop=individ_pop)
    evolved_pop = reproduction(individ_pop, crossingover, probability)
    update_pop_fitness(evolved_pop, fitness_func, environment)
    return evolved_pop
