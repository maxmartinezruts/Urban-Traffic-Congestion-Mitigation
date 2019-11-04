# Author:   Max Martinez Ruts
# Creation: 2019

import tensorflow as tf
import numpy as np


x = np.zeros((100000, 16), dtype=float)
y = np.zeros((100000, 16), dtype=float)


for i in range(0, 100000):
    for j in range(0, 4):
        time = np.random.uniform(0, 3000)
        index = np.random.randint(0,4)
        x[i,4 * j + index] = time/3000
        if time < 2000:
            y[i, 4 * j + ((index + 0) % 4)] = 1.0
        else:
            y[i, 4 * j + ((index + 1) % 4)] = 1.0
stages = np.random.randint(0, 3, (100000, 4))

x_train = x[:90000]
x_test = x[90000:]

y_train = y[:90000]
y_test = y[90000:]

model_base = tf.keras.models.Sequential()

# Add input, hidden and output layers
input_layer = tf.keras.layers.Flatten()
hidden_layer = tf.keras.layers.Dense(units=100, input_shape=[16], activation='relu')
hidden_layer2 = tf.keras.layers.Dense(units=100, input_shape=[100], activation='relu')
hidden_layer3 = tf.keras.layers.Dense(units=100, input_shape=[100], activation='relu')


output_layer = tf.keras.layers.Dense(units=16, input_shape=[100], activation='sigmoid')

model_base.add(input_layer)
model_base.add(hidden_layer)
model_base.add(hidden_layer2)
model_base.add(hidden_layer3)

model_base.add(output_layer)

model_base.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# model_base.fit(x_train,y_train, epochs=50, batch_size=128)
# model_base.save_weights('weights_3h')
weights = model_base.load_weights('weights_3h')
pr =[0.,         0.,         0.68,        0. ,        0.,         0.,
  0.,         0.1,        0. ,        0.,         0.16666667, 0.,
  0.,         0.13333333, 0.,         0.,        ]
ip = np.array([pr])
input = np.asarray([list(ip)])
# Initialize template ANN
ip = np.array([0,0.26666666,0.,0.,0.,0.,0.7333333,0,0.7666666666,0.,0.,0,0.7,0.,0.,0.],dtype=float)
input = np.asarray([list(ip)])

print(ip)
print('seee',model_base.predict(input, verbose=0))


y = model_base.predict(x_test)
print(y[0])
for i in range(0,100):
    print('-------')
    print(np.round(y[i]),y_test[i],x_test[i])





# Class containing all information of a vehicle controller (ANN)
class Brain:
    def __init__(self, model):

        # If model not yet specified, create a new model
        if model==0:
            # self.model =  tf.keras.models.clone_model(model_base)
            # self.model.set_weights( model_base.get_weights())

            self.model = tf.keras.models.clone_model(model_base)
            self.model.build((None, 16))  # replace 10 with number of variables in input layer
            # self.model.compile(optimizer='rmsprop',
            #                    loss='mean_squared_error',
            #                    metrics=['accuracy'])
            self.model.set_weights(model_base.get_weights())

            ip = np.array([0, 0.26666666, 0., 0., 0., 0., 0.7333333, 0, 0.7666666666, 0., 0., 0, 0.7, 0., 0., 0.],
                          dtype=float)

            input = np.asarray([list(np.array(ip,dtype=float))])
            print('Output:',self.model.predict(input,1)[0])
        # Otherwise use the passed model
        else:
            self.model = tf.keras.models.clone_model(model, input_tensors=None)

    # Execute crossover by mixing 2 genotypes
    def crossover(self, genes_1, genes_2):

        # Generate hybrid copies from progenitors
        weights_hidden = (genes_1[0]+genes_2[0])/2
        biases_hidden = (genes_1[1]+genes_2[1])/2

        weights_hidden2 = (genes_1[2] + genes_2[2]) / 2
        biases_hidden2 = (genes_1[3] + genes_2[3]) / 2

        weights_hidden3 = (genes_1[4] + genes_2[4]) / 2
        biases_hidden3 = (genes_1[5] + genes_2[5]) / 2

        weights_outputs = (genes_1[6]+genes_2[6])/2
        biases_outputs = (genes_1[7]+genes_2[7])/2

        # Set genotype
        self.weights = [weights_hidden, biases_hidden,weights_hidden2,biases_hidden2,weights_hidden3, biases_hidden3,weights_outputs, biases_outputs]

        # Update weights and biases
        self.model.set_weights(self.weights)

    def mutate(self):
        self.weights = self.model.get_weights()

        factor = 10
        mutation_rate = 0.99
        # Create random matrices and add only random values to elements in the matrix where value specified is greater than mutation rate
        # Mutation magnitude is set equal to random normal value
        w1 = np.random.randn(16,100)
        r = np.random.rand(16,100)
        w1 = np.where(r>mutation_rate,w1,0)

        b1 = np.random.randn(100)
        r = np.random.rand(100)
        b1 = np.where(r> mutation_rate, b1, 0)

        w2 = np.random.randn(100, 100)
        r = np.random.rand(100, 100)
        w2 = np.where(r > mutation_rate, w2, 0)

        b2 = np.random.randn(100) / 2
        r = np.random.rand(100)
        b2 = np.where(r > mutation_rate, b2, 0)

        w3 = np.random.randn(100, 100)
        r = np.random.rand(100, 100)
        w3 = np.where(r > mutation_rate, w3, 0)

        b3 = np.random.randn(100) / 2
        r = np.random.rand(100)
        b3 = np.where(r > mutation_rate, b3, 0)

        w4 = np.random.randn(100, 16)
        r = np.random.rand(100, 16)
        w4 = np.where(r > mutation_rate, w4, 0)

        b4 = np.random.randn(16) / 2
        r = np.random.rand(16)
        b4 = np.where(r > mutation_rate, b4, 0)
        # Add mutation matrices to existing genotype
        self.weights[0] += w1/factor
        self.weights[1] += b1/factor
        self.weights[2] += w2/factor
        self.weights[3] += b2/factor

        self.weights[4] += w3/factor
        self.weights[5] += b3/factor
        self.weights[6] += w4/factor
        self.weights[7] += b4/factor

        # Update weights and biases
        self.model.set_weights(self.weights)

    # Set weights and biases of ANN
    def create(self):
        self.model.set_weights(self.weights)
