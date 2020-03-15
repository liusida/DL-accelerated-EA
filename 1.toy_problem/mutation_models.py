# mutation models
import numpy as np

from helper import *

# base class
class mutation_model:
    def __init__(self, evolution_rate):
        self.evolution_rate = evolution_rate
        pass
    def train(self, history):
        """ history = {(x0,x1): y} """
        pass
    def mutate(self, population):
        pass

# toy model
class toy_model(mutation_model):
    def __init__(self, evolution_rate):
        super().__init__(evolution_rate)
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        self.keras_model = Sequential()
        self.keras_model.add(Dense(32, input_dim=2))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(32))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(1, activation='sigmoid'))
        self.keras_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    def _train(self, x, y):
        pass
    def train(self,history):
        X = []
        Y = []
        for key in history.keys():
            x0 = key[0]
            x1 = key[1]
            y = history[key]
            X.append([x0,x1])
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        self.keras_model.fit(X, Y, batch_size=32, epochs=1)
    def mutate(self, population):
        #TODO: implement keras-custom-training-loop
        # https://towardsdatascience.com/keras-custom-training-loop-59ce779d60fb
        return population

# baseline random model
class random_model(mutation_model):
    def mutate(self, population):
        mutant = empty_like(population)
        for i in range(size(population)):
            for key in population.keys():
                mutant[key].append(population[key][i] + (np.random.random()-0.5)*self.evolution_rate)
        return mutant
