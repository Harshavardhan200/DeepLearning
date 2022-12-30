from keras.layers import Dense
from numpy import array
from random import seed, randint
from fractions import Fraction
from keras.models import Sequential, load_model
import keras.models as mode
import pandas as pd
class MyNN_add_two_numbers:
    def __init__(self, valueCount=100, maxValue=100):
        self.count = valueCount
        
        self.maxvalue = maxValue
        self.model = Sequential()
    def denormalize(self, value, maxvalue):
        return value * float(maxvalue * 2.0)
    def normalize(self, value, maxvalue):
        return value.astype('float') / float(maxvalue * 2.0)
    def setup_model(self):
        self.model.add(Dense(4, input_dim=2)) #hidden layer with 4 nodes
        self.model.add(Dense(2))     #hidden layer with 2 nodes
        self.model.add(Dense(1))     #hidden layer with 2 nodes
        self.model.compile(loss='mean_squared_error', optimizer='adam') #compile the model
    def create_datasets(self, count, max_value):
        addends = list()
        sums = list()
        for n in range(count):
            addends.append([randint(0, max_value), randint(0, max_value)])
            sums.append(sum(addends[n]))
        addends = array(addends)
        sums = array(sums)
        addends = self.normalize(addends, self.maxvalue)
        sums = self.normalize(sums, self.maxvalue)
        return addends, sums
    def train_model(self, number_of_times=50, epochs=3, verbose=0, batch_size=2):
        for _ in range(number_of_times):
            addends, sums = self.create_datasets(self.count, self.maxvalue)
            self.model.fit(addends, sums, epochs=epochs, verbose=verbose, batch_size=2)
    def save_model(self, fileName):
        self.model.save(fileName)
    def predict_output(self, input_array):
        input_array1 = self.normalize(input_array, self.maxvalue)
        test = self.model.predict(input_array1, batch_size=1, verbose=0)
        for i in range(len(test)):
            addend = self.denormalize(input_array[i][0], self.maxvalue)
            augend = self.denormalize(input_array[i][1], self.maxvalue)
            total = self.denormalize(test[i][0], self.maxvalue)
            print('{:4d} {:12.6f} {:12.6f} {:12.6f} {:8.2f}'.format(i, addend, augend, total, abs(total)))
model = MyNN_add_two_numbers()
model.setup_model()
model.train_model()
model.predict_output(input_array= array([[1200, 1343], [1, 1], [-3, -3], [Fraction(16, 5), 3.25]]))      