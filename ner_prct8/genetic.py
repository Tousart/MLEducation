from deap import base, creator, tools
from network import NNetwork
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import algelitism

np.bool8 = bool

env = gym.make('CartPole-v1')

NEURONS_IN_LAYERS = [4, 1]
net = NNetwork(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NNetwork.getTotalWeights(*NEURONS_IN_LAYERS)
LOW = -0.1
UP = 1.0
ETA = 20

# константы генетического алгоритма
POPULATION_SIZE = 20
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 2

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# класс индивида
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# регистрируем функции
toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

# функция для вычисления приспособленности особи
def get_score(individual):
    net.set_weights(individual)

    observation, _ = env.reset()
    action_counter = 0
    total_reward = 0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action_counter += 1
        action = int(net.predict(observation.reshape(1, -1)))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    return total_reward, 


# регистрируем еще парочку функций)
toolbox.register("evaluate", get_score)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          halloffame=hof,
                                          stats=stats,
                                          verbose=True)

max_fitness_values, mean_fitness_values = logbook.select("max", "avg")

best = hof.items[0]
print(best)

plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Поколение')
plt.ylabel('Максимальная/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

