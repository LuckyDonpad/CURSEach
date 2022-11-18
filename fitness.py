# принимает модель нейросети model, объект класса среды обитания environment.py
# возвращает приспособленность индивида к заданной среде
def get_model_fitness(model, environment):
    m_history = model.fit(verbose=1, x=environment.train_x, y=environment.train_y,
                          validation_data=(environment.val_x, environment.val_y),
                          batch_size=32, epochs=100)
    return max(m_history.history['val_accuracy'])

# обновляет параметр приспособленности каждого индивида в популяции individ_pop
# высчитываемый при помощи функции fitness_function в среде environment.py
def update_pop_fitness(individ_pop, fitness_function, environment):
    for individ in individ_pop:
        if individ.fitness == 0:
            individ.update_fitness(fitness_function=fitness_function,
                                   environment=environment)
