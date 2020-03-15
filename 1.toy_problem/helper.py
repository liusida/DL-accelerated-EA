# some helper functions
def size(population):
    anykey = list(population.keys())[0]
    return len(population[anykey])
def empty_like(population):
    ret = {}
    for key in population.keys():
        ret[key] = []
    return ret
