from deap import tools
from deap.algorithms import varAnd

# переделанный алгоритм eaSimple с элементом элитизма
def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
        halloffame=None, verbose =__debug__ , callback=None):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # оценка слабых особей
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # выбор особей следующего поколения
        offspring = toolbox.select(population, len(population) - hof_size)
        # print(offspring)
        # изменение набора особей
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # оценка слабых особей
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # обновление вознаграждений
        if halloffame is not None:
            halloffame.update(offspring)

        # замена текущей популяции потомством
        population[:] = offspring

        # добавление статистики генерации в журнал регистрации
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        if callback:
            callback[0](*callback[1])

        return population, logbook