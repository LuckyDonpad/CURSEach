import pandas as pd
import functools
from utils import *
from environment import Environment
import shapes

if __name__ == '__main__':
    print(globals())

    shapes.INPUT_SHAPE = (9,)  # формат входных данных
    shapes.OUT_CLASSES = 2
    # предобработка датасета
    dataset_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    dataset = pd.read_csv(dataset_url)
    dataset = dataset.drop(labels=['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
    dataset["Age"] = dataset['Age'].fillna(int(dataset["Age"].sum() / len(dataset["Age"])))
    dataset["Embarked"] = dataset["Embarked"].fillna('S')
    dataset["Sex"] = dataset["Sex"].map({'male': 0, "female": 1})
    dataset = pd.get_dummies(dataset, columns=['Embarked'])

    # разбиение датасета на тестовый и валидационный датасеты
    train = dataset.sample(frac=0.7, random_state=42)
    test = dataset.drop(train.index)
    train_y = np.array(train['Survived'])
    train_x = np.array(train.drop('Survived', axis='columns'))
    test_y = np.array(test['Survived'])
    test_x = np.array(test.drop('Survived', axis='columns'))

    # подготовка всех участников эволюции
    gen_pop = generate_genom_population(20)
    model_pop = generate_models_population(gen_pop)
    individ_pop = generate_individ_population(gen_pop, model_pop)
    environment = Environment(train_x, train_y, test_x, test_y)

    for _ in range(30):
        individ_pop = evolve(individ_pop=individ_pop, fitness_func=get_model_fitness,
                             environment=environment, crossingover=crossingover, probability=1 / 3)
        stats = []
        for individ in individ_pop:
            stats.append([individ.genom.layers, individ.genom.neurons, individ.fitness])
        stats.sort(key=functools.cmp_to_key(lambda x, y: y[-1] - x[-1]))
        print(stats)
