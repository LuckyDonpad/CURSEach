import tensorflow as tf
from genom import Genom
from individ import Individ
import numpy as np
import random
import shapes
import tournament


# принимает: геном нейросети - genom - объект класса Genom,
# inp_shape - кортеж описывающий форму входных данных
# out_classes - количество классов для прогноза
# возвращает скомпилированную модель нейросети
def generate_neural_network_from_genom(genom, inp_shape, out_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=inp_shape[0], input_shape=inp_shape))
    for layer in genom.neurons:
        model.add(tf.keras.layers.Dense(layer))
    model.add(tf.keras.layers.Dense(out_classes, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# возвращает массив размера size случайных геномов
def generate_genom_population(size):
    population = []
    for _ in range(size):
        gen = Genom([])
        gen.random_genom()
        population.append(gen)
    return population


# возвращает массив нейросетей сгенерированных из массива геномов genom_list
def generate_models_population(genom_list):
    population = []
    for genom in genom_list:
        individ = generate_neural_network_from_genom(genom, shapes.INPUT_SHAPE, shapes.OUT_CLASSES)
        population.append(individ)
    return population


# принимает модель нейросети model, объект класса среды обитания environment.py
# возвращает приспособленность индивида к заданной среде
def get_model_fitness(model, environment):
    m_history = model.fit(verbose=1, x=environment.train_x, y=environment.train_y,
                          validation_data=(environment.val_x, environment.val_y),
                          batch_size=32, epochs=100)
    return max(m_history.history['val_accuracy'])


# принимает в себя массив геномов - genoms, массив моделей - models
# возвращает массив объектов класса Individ
def generate_individ_population(genoms, models):
    individ_population = []
    for i in range(len(genoms)):
        individ = Individ(genoms[i], models[i])
        individ_population.append(individ)
    return individ_population


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
