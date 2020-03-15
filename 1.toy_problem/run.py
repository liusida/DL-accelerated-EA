import matplotlib.pyplot as plt
import random
import numpy as np

from helper import *
from mutation_models import *

# toy problem setting
random.seed(2)
np.random.seed(2)
population_size = 4
desired_point = {"x0":np.random.random()*100, "x1":np.random.random()*100}
evolution_rate = 1
total_generation = 100
full_history = {}
mutation_models = [toy_model(evolution_rate), random_model(evolution_rate)]

# compare the behaviors of two mutation models
for mutation_model in mutation_models:
    random.seed(1)
    np.random.seed(1)

    # randomly init population
    population = {"x0": [], "x1": []}
    for i in range(population_size):
        population["x0"].append(np.random.random()*100)
        population["x1"].append(np.random.random()*100)

    generation = 0
    plt.scatter(population["x0"], population["x1"], c="blue", s=80, label="start position")
    for i in range(total_generation):
        # evaluation
        results = []
        for i in range(size(population)):
            ret = -np.sqrt((population["x0"][i] - desired_point["x0"])**2 + (population["x1"][i] - desired_point["x1"])**2)
            results.append(ret)
            full_history[(population["x0"][i], population["x1"][i])] = ret
        mutation_model.train(full_history)

        sorted_index = np.argsort(results)[::-1]
        # selection
        half = int(len(results)/2)
        selected = empty_like(population)
        for i in range(half):
            for key in population.keys():
                selected[key].append( population[key][sorted_index[i]] )
        # mutation
        mutant1 = mutation_model.mutate(selected)
        mutant2 = mutation_model.mutate(selected)
        
        # next generation
        next_generation = empty_like(population)
        for i in range(half):
            for key in population.keys():
                next_generation[key].append( mutant1[key][i] )
                next_generation[key].append( mutant2[key][i] )

        plt.scatter(next_generation["x0"], next_generation["x1"], c=f"{0.8*(1-generation/total_generation)}", s=1)
        population = next_generation
        generation += 1
        
    plt.scatter(next_generation["x0"], next_generation["x1"], c="green", s=80, label="final position")

    plt.scatter(desired_point["x0"], desired_point["x1"], c="red", s=80, label="desired point")
    plt.legend()
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()
