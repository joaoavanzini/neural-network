# coding=<ASCII>
# python 2.7
# machine learning, neural networks
# features - two properties: how time the person sleep + how time the person study, then that results note of your test
# features 8 + 4 -> 7
# features 0.8 + 0.4 -> 0.7

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(2, 1) # input of 2 (elements: 0.8 and 0.4) elements, and output 1 element (element: 0.7)

# adds the parameters
ds.addSample((0.8, 0.4), (0.7))
ds.addSample((0.5, 0.7), (0.5))
ds.addSample((1.0, 0.8), (0.95))

# creates the neural network
nn = buildNetwork(2, 4, 1, bias=True) # A network architecture: 2 input neurons, 4 neurons in the hidden layer and 1 neuron in the output layer, and places the bias as true - neural networks with bias generally learn faster

# creates the trainer
trainer = BackpropTrainer(nn, ds) # as arguments, we place the neural network plus the dataset (the parameters)

# creates interaction
# xrange is for python 2.7
# trains the neural network
for i in xrange(2000):
	print(trainer.train())
# training of the neural network. The closer to 0, the better!
# 2000 is the number of times it will be tested


while True:
	sleep = float(raw_input('Sleep: \n')) # float value for the exit, ie: will ask how long the person slept
	study = float(raw_input('Study: \n')) # float value for output, ie: will ask how long the person studied

	z = nn.activate((sleep, study))[0] * 10.0 # activate is used to compute the values. This line serves to show if the person slept, or studied. The array at the end serves to get the first and only digit that the neural network creates. And, after being multiplied by 10, so that it is an equal measure of proof

	print('Accuracy of note: ', str(z)) # can predict the score according to the amount of hours spent and hours studied - shows this forecast

