import random


# возвращает наиболее приспособленного индивида из двух ind_1, ind_2
def binary_tournament(ind_1, ind_2):
    return ind_1 if ind_1.fitness > ind_2.fitness else ind_2


# производит турнирный отбор популяции individ_pop
# возвращает новую популяцию из тех кто прошел отбор
# половинного размера от исходной
def tournament_selection(individ_pop):
    new_pop = []
    for i in range(len(individ_pop) // 2):
        index_1 = random.randint(0, len(individ_pop) - 1)
        index_2 = random.randint(0, len(individ_pop) - 1)
        while index_1 == index_2:
            index_2 = random.randint(0, len(individ_pop) - 1)
        new_pop.append(binary_tournament(individ_pop[index_1], individ_pop[index_2]))
    return new_pop
