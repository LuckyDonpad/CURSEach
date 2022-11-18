import tensorflow as tf
from genom import Genom
import shapes
from individ import Individ


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


# принимает в себя массив геномов - genoms, массив моделей - models
# возвращает массив объектов класса Individ
def generate_individ_population(genoms, models):
    individ_population = []
    for i in range(len(genoms)):
        individ = Individ(genoms[i], models[i])
        individ_population.append(individ)
    return individ_population
