# import keras
import matplotlib.pyplot as plt
import random
import numpy as np

# some helper functions
def size(population):
    anykey = list(population.keys())[0]
    return len(population[anykey])
def empty_like(population):
    ret = {}
    for key in population.keys():
        ret[key] = []
    return ret

# mutation models
class mutation_model:
    def __init__(self):
        pass
    def train(self, history):
        pass
    def mutate(self, population):
        pass
class toy_model(mutation_model):
    def _train(self, x, y):
        pass
class random_model(mutation_model):
    def mutate(self, population):
        mutant = empty_like(population)
        for i in range(size(population)):
            for key in population.keys():
                mutant[key].append(population[key][i] + (np.random.random()-0.5)*evolution_rate)
        return mutant

# toy problem setting
random.seed(2)
np.random.seed(2)
population_size = 4
desired_point = {"x":np.random.random()*100, "y":np.random.random()*100}
evolution_rate = 1
total_generation = 100
full_history = {}
mutation_models = [random_model(), random_model()]

# compare the behaviors of two mutation models
for mutation_model in mutation_models:
    random.seed(1)
    np.random.seed(1)

    # randomly init population
    population = {"x": [], "y": []}
    for i in range(population_size):
        population["x"].append(np.random.random()*100)
        population["y"].append(np.random.random()*100)

    generation = 0
    plt.scatter(population["x"], population["y"], c="blue", s=80, label="start position")
    for i in range(total_generation):
        # evaluation
        results = []
        for i in range(size(population)):
            ret = -np.sqrt((population["x"][i] - desired_point["x"])**2 + (population["y"][i] - desired_point["y"])**2)
            results.append(ret)
            full_history[(population["x"][i], population["y"][i])] = ret
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

        plt.scatter(next_generation["x"], next_generation["y"], c=f"{0.8*(1-generation/total_generation)}", s=1)
        population = next_generation
        generation += 1
        
    plt.scatter(next_generation["x"], next_generation["y"], c="green", s=80, label="final position")

    plt.scatter(desired_point["x"], desired_point["y"], c="red", s=80, label="desired point")
    plt.legend()
    plt.show()
