from data import load_data, array2midi, one_hot_decoder
# from aux import _path
from grulayer import GRUOutputInLayer
import cPickle as pickle
import os.path
from os import listdir

import lasagne
from lasagne.layers import (
    InputLayer, DenseLayer, GRULayer, ConcatLayer, SliceLayer, ReshapeLayer,
    get_output, get_all_params, get_all_param_values, set_all_param_values
)

import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks', palette='Set2')

####### RNN Models for folk music composition ########
class MusicModelGRU(object):
	"""docstring for MusicModelGRU"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25, use_deterministic_previous_output=True):
		#super(MusicModelGRU, self).__init__()
		
		# Model naming and data path
		self.model_name = model_name
		self.model_data_path = "../data/models/"

		# Model hyperparameters
		self.num_gru_layer_units = num_gru_layer_units
		self.use_deterministic_previous_output = use_deterministic_previous_output
		
		# Model feature dimensions
		self.max_seq_len = max_seq_len-1
		self.num_features_pitch = num_features_pitch
		self.num_features_duration = num_features_duration
		
		# Training metadata
		self.batch_size = 10
		self.number_of_epochs_trained = 0

		### Learning curve data ### 
		# Categorical Crossentropy Costs 
		self.cost_train_pitch = []
		self.cost_train_duration = []
		self.cost_valid_pitch = []
		self.cost_valid_duration = []
		
		# Accuracies
		self.acc_train_pitch = []
		self.acc_train_duration = []
		self.acc_valid_pitch = []
		self.acc_valid_duration = []

		# Norms over horizontal GRU weights
		self.horz_update = []
		self.horz_reset = []
		self.horz_hidden = []

		# Norms over vertical GRU weights
		self.vert_update = []
		self.vert_reset = []
		self.vert_hidden = []

		### symbolic theano variables ### 
		# Note that we are using itensor3 as we 3D one-hot-encoded input (integers)
		self.x_pitch_sym = T.itensor3('x_pitch_sym')
		self.x_duration_sym = T.itensor3('x_duration_sym')

		self.y_pitch_sym = T.itensor3('y_pitch_sym')
		self.y_duration_sym = T.itensor3('y_duration_sym')

		self.mask_sym = T.matrix('mask_sym')
		
	
	def train(self, train_data, valid_data=None, num_epochs=10, batch_size=None):
		# This generative model takes in the: 
			# original data: x = {"pitch": x_pitch, "duration: x_duration, "mask": x_mask] (a list of 3 numpy arrays) with dimensions:
				# dim(x_pitch) = (N, max_seq_len, num_features_pitch)
				# dim(x_duration) = (N, max_seq_len, num_features_duration)
				# dim(x_mask) = (N, max_seq_len)
			# The original data is reformed into: 
				# input: x_inpu = [x_{0}, ..., x_{max_seq_len-1}] with dim(x)=(N, max_seq_len-1, num_features)
				# target: y = [x_{1}, ..., x_{max_seq_len}] with dim(y)=(N, max_seq_len-1, num_features) - this is one-step-ahead target that the model will predict.
		if batch_size is not None:
			self.batch_size = batch_size


		### COLLECT AND SPLIT DATA ###
		print("Reforming data into inputs (x) and targets (y):")
		# Train data
		x_pitch_train, y_pitch_train, x_duration_train, y_duration_train, mask_train = data_setup(train_data)	

		# Validation data
		if valid_data is not None:
			x_pitch_valid, y_pitch_valid, x_duration_valid, y_duration_valid, mask_valid = data_setup(valid_data)	

		### TRAINING ###
		print("Training model on given data for {} epochs:".format(num_epochs))
		N_train = x_pitch_train.shape[0]

		header_string = "Cost:\tPitch\tDuration| Acc:\tPitch\tDuration"
		valid_string = ""
		for epoch in range(self.number_of_epochs_trained, self.number_of_epochs_trained + num_epochs):
			shuffled_indices = np.random.permutation(N_train)
			for i in range(0, N_train, self.batch_size):
				# Collect random batch 
				subset = shuffled_indices[i:(i + self.batch_size)]
				x_pitch_batch = x_pitch_train[subset]
				y_pitch_batch = y_pitch_train[subset]
				x_duration_batch = x_duration_train[subset]
				y_duration_batch = y_duration_train[subset]
				mask_batch = mask_train[subset]
				# Train for batch and collect cost, accuracy and output
				self.f_train(x_pitch_batch, y_pitch_batch, x_duration_batch, y_duration_batch, mask_batch)
				# epoch_cost += batch_cost
			train_cost_pitch, train_acc_pitch, train_output_pitch, train_cost_duration, train_acc_duration, train_output_duration = self.f_eval(x_pitch_train, y_pitch_train, x_duration_train, y_duration_train, mask_train)
			train_string = "Train: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(train_cost_pitch), float(train_cost_duration), float(train_acc_pitch), float(train_acc_duration))

			# Compute norms over horizontal GRU weights
			self.horz_update += [np.linalg.norm(self.l_out_gru.W_hid_to_updategate.get_value())]
			self.horz_reset += [np.linalg.norm(self.l_out_gru.W_hid_to_resetgate.get_value())]
			self.horz_hidden += [np.linalg.norm(self.l_out_gru.W_hid_to_hidden_update.get_value())]

			# Compute norms over vertical GRU weights
			self.vert_update += [np.linalg.norm(self.l_out_gru.W_in_to_updategate.get_value())]
			self.vert_reset += [np.linalg.norm(self.l_out_gru.W_in_to_resetgate.get_value())]
			self.vert_hidden += [np.linalg.norm(self.l_out_gru.W_in_to_hidden_update.get_value())]

			self.cost_train_pitch += [train_cost_pitch]
			self.acc_train_pitch += [train_acc_pitch]
			self.cost_train_duration += [train_cost_duration]
			self.acc_train_duration += [train_acc_duration]

			if valid_data is not None:
				valid_cost_pitch, valid_acc_pitch, valid_output_pitch, valid_cost_duration, valid_acc_duration, valid_output_duration = self.f_eval(x_pitch_valid, y_pitch_valid, x_duration_valid, y_duration_valid, mask_valid)
				self.cost_valid_pitch += [valid_cost_pitch]
				self.acc_valid_pitch += [valid_acc_pitch]
				self.cost_valid_duration += [valid_cost_duration]
				self.acc_valid_duration += [valid_acc_duration]
				valid_string = "Valid: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(valid_cost_pitch), float(valid_cost_duration), float(valid_acc_pitch), float(valid_acc_duration))
			
			epoch_string = "\nEpoch {:2d}: {}\n{}\n{}".format(epoch + 1, header_string, train_string, valid_string)
			print(epoch_string)

		# Update the number of epochs the model have been trained for
		self.number_of_epochs_trained += num_epochs


	def evaluate(self, data, write2midi=False, pitch_map=None, duration_map=None):
		# Model reconstructions on the given evaluation data:
			# original data: x = {"pitch": x_pitch, "duration: x_duration, "mask": x_mask] (a list of 3 numpy arrays) with dimensions:
				# dim(x_pitch) = (N, max_seq_len, num_features_pitch)
				# dim(x_duration) = (N, max_seq_len, num_features_duration)
				# dim(x_mask) = (N, max_seq_len)
			# The original data is reformed into: 
				# input: x_inpu = [x_{0}, ..., x_{max_seq_len-1}] with dim(x)=(N, max_seq_len-1, num_features)
				# target: y = [x_{1}, ..., x_{max_seq_len}] with dim(y)=(N, max_seq_len-1, num_features) - this is one-step-ahead target that the model will predict.
		x_pitch, y_pitch, x_duration, y_duration, mask = data_setup(data)	

		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = self.f_eval(x_pitch, y_pitch, x_duration, y_duration, mask)

		if write2midi:
			filename = self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained)
			metadata = data["metadata"]
			indices = data["indices"]
			array2midi(y_pitch, pitch_map, y_duration, duration_map, metadata, indices, filepath=self.model_data_path, filename=filename + "original")
			array2midi(output_pitch, pitch_map, output_duration, duration_map, metadata, indices, filepath=self.model_data_path, filename=filename + "reconstruction")

		return cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration


	def save(self):
		model_path = self.model_data_path + self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained) + ".pkl"
		print("Saving present model parameters, metadata and learning curves in {}".format(model_path))
		### SAVE model ###
		model = {}

		# Hyperparameters
		model["max_seq_len"] = self.max_seq_len
		model["num_features_pitch"] = self.num_features_pitch
		model["num_features_duration"] = self.num_features_duration
		model["num_gru_layer_units"] = self.num_gru_layer_units
		model["number_of_epochs_trained"] = self.number_of_epochs_trained
		print("Saving number_of_epochs_trained = {}".format(self.number_of_epochs_trained))
		model["use_deterministic_previous_output"] = self.use_deterministic_previous_output

		# Parameters
		model["parameters"] = get_all_param_values([self.l_out_pitch, self.l_out_duration])

		# Costs
		model["cost_train_pitch"] = self.cost_train_pitch
		model["cost_valid_pitch"] = self.cost_valid_pitch
		model["cost_train_duration"] = self.cost_train_duration
		model["cost_valid_duration"] = self.cost_valid_duration

		# Accuracies
		model["acc_train_pitch"] = self.acc_train_pitch
		model["acc_valid_pitch"] = self.acc_valid_pitch
		model["acc_train_duration"] = self.acc_train_duration
		model["acc_valid_duration"] = self.acc_valid_duration

		# Compute norms over horizontal GRU weights
		model["horz_update"] = self.horz_update
		model["horz_reset"] = self.horz_reset
		model["horz_hidden"] = self.horz_hidden

		# Compute norms over vertical GRU weights
		model["vert_update"] = self.vert_update
		model["vert_reset"] = self.vert_reset
		model["vert_hidden"] = self.vert_hidden

		with open(model_path, "wb") as file:
			pickle.dump(model, file)

	def load(self, model_name, number_of_epochs_trained=None):
		model_loaded = False
		### LOAD model ###
		model_name_spec = model_name + "_gru_{}_bs_{}_e_".format(self.num_gru_layer_units, self.batch_size)

		if number_of_epochs_trained is None:
			model_epochs = [int(file.split(".")[0].split("_")[-1]) for file in listdir(self.model_data_path) if (file[0] != "." and file[:len(model_name_spec)] == model_name_spec and file.split(".")[-1] == "pkl")]
		else: 
			model_epochs = [number_of_epochs_trained]

		# Check for latest or the number_of_epochs_trained model data
		if model_epochs:
			max_epoch_num = max(model_epochs)
			print("The current number of epochs the {} model have been trained is: {}".format(model_name, max_epoch_num))
			print("Loading the data for the current state of the model.")
			model_path = self.model_data_path + model_name_spec + str(max_epoch_num) + ".pkl"
			print("Will load {}".format(model_path))
			if os.path.isfile(model_path):
				self.model_name = model_name
				with open(model_path, "rb") as file:
					model = pickle.load(file)
				model_loaded = True
				print("Loaded {}".format(model_path))
		else: 
			print("No previous data on this model exists. Use the methods train() and save() first and then load().")

		if model_loaded:
			print("Setting up model with previous parameters from the file {}".format(model_path))

			# Hyperparameters
			self.max_seq_len = model["max_seq_len"]
			self.num_features_pitch = model["num_features_pitch"]
			self.num_features_duration = model["num_features_duration"]
			self.num_gru_layer_units = model["num_gru_layer_units"]
			self.number_of_epochs_trained = model["number_of_epochs_trained"]
			self.use_deterministic_previous_output = model["use_deterministic_previous_output"]

			# Parameters
			set_all_param_values([self.l_out_pitch, self.l_out_duration], model["parameters"])

			# Costs \
			self.cost_train_pitch = model["cost_train_pitch"]
			self.cost_valid_pitch = model["cost_valid_pitch"]
			self.cost_train_duration = model["cost_train_duration"]
			self.cost_valid_duration = model["cost_valid_duration"]

			# Accuracies
			self.acc_train_pitch = model["acc_train_pitch"]
			self.acc_valid_pitch = model["acc_valid_pitch"]
			self.acc_train_duration = model["acc_train_duration"]
			self.acc_valid_duration = model["acc_valid_duration"]

			# Compute norms over horizontal GRU weights
			self.horz_update = model["horz_update"]
			self.horz_reset = model["horz_reset"]
			self.horz_hidden = model["horz_hidden"]

			# Compute norms over vertical GRU weights
			self.vert_update = model["vert_update"]
			self.vert_reset = model["vert_reset"]
			self.vert_hidden = model["vert_hidden"]

	def plotLearningCurves(self):
		model_path = self.model_data_path + self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained)
		epochs = range(1, self.number_of_epochs_trained+1)

		# Accuracy plots
		plt.figure()
		acc_train_pitch_plt, = plt.plot(epochs, self.acc_train_pitch, 'r-')
		acc_valid_pitch_plt, = plt.plot(epochs, self.acc_valid_pitch, 'r--')
		acc_train_duration_plt, = plt.plot(epochs, self.acc_train_duration, 'b-')
		acc_valid_duration_plt, = plt.plot(epochs, self.acc_valid_duration, 'b--')
		plt.ylabel('Accuracies', fontsize=15)
		plt.xlabel('Epoch #', fontsize=15)
		plt.legend([acc_train_pitch_plt, acc_valid_pitch_plt, acc_train_duration_plt, acc_valid_duration_plt], ["Training Pitch", "Validation Pitch", "Training Duration", "Validation Duration"])
		plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_acc.png")

		# Cost plots
		plt.figure()
		cost_train_pitch_plt, = plt.plot(epochs, self.cost_train_pitch, 'r-')
		cost_valid_pitch_plt, = plt.plot(epochs, self.cost_valid_pitch, 'r--')
		cost_train_duration_plt, = plt.plot(epochs, self.cost_train_duration, 'b-')
		cost_valid_duration_plt, = plt.plot(epochs, self.cost_valid_duration, 'b--')
		plt.ylabel('Crossentropy Costs', fontsize=15)
		plt.xlabel('Epoch #', fontsize=15)
		plt.legend([cost_train_pitch_plt, cost_valid_pitch_plt, cost_train_duration_plt, cost_valid_duration_plt], ["Training Pitch", "Validation Pitch", "Training Duration", "Validation Duration"])
		plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_cost.png")

		# Horizontal weights plots
		plt.figure()
		horz_update_plt, = plt.plot(epochs, self.horz_update)
		horz_reset_plt, = plt.plot(epochs, self.horz_reset)
		horz_hidden_plt, = plt.plot(epochs, self.horz_hidden)
		plt.ylabel('Frobenius Norm of Horizontal Weights', fontsize=15)
		plt.xlabel('Epoch #', fontsize=15)
		plt.legend([horz_update_plt, horz_reset_plt, horz_hidden_plt], ["Update Gate", "Reset Gate", "Hidden Update Gate"])
		plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_horzWeights.png")

		# Vertical weights plots
		plt.figure()
		vert_update_plt, = plt.plot(epochs, self.vert_update)
		vert_reset_plt, = plt.plot(epochs, self.vert_reset)
		vert_hidden_plt, = plt.plot(epochs, self.vert_hidden)
		plt.ylabel('Frobenius Norm of Vertical Weights', fontsize=15)
		plt.xlabel('Epoch #', fontsize=15)
		plt.legend([vert_update_plt, vert_reset_plt, vert_hidden_plt], ["Update Gate", "Reset Gate", "Hidden Update Gate"])
		plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_vertWeights.png")

class GRU_Network_Using_Previous_Output(MusicModelGRU):
	"""docstring for GRU_Network_Using_Previous_Output"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25, use_deterministic_previous_output=True):
		# super(GRU_Network_Using_Previous_Output, self).__init__()
		
		# Model naming and data path
		self.model_name = model_name
		self.model_data_path = "../data/models/"

		# Model hyperparameters
		self.num_gru_layer_units = num_gru_layer_units
		self.use_deterministic_previous_output = use_deterministic_previous_output
		
		# Model feature dimensions
		self.max_seq_len = max_seq_len-1
		self.num_features_pitch = num_features_pitch
		self.num_features_duration = num_features_duration
		
		# Training metadata
		self.batch_size = 10
		self.number_of_epochs_trained = 0

		### Learning curve data ### 
		# Categorical Crossentropy Costs 
		self.cost_train_pitch = []
		self.cost_train_duration = []
		self.cost_valid_pitch = []
		self.cost_valid_duration = []
		
		# Accuracies
		self.acc_train_pitch = []
		self.acc_train_duration = []
		self.acc_valid_pitch = []
		self.acc_valid_duration = []

		# Norms over horizontal GRU weights
		self.horz_update = []
		self.horz_reset = []
		self.horz_hidden = []

		# Norms over vertical GRU weights
		self.vert_update = []
		self.vert_reset = []
		self.vert_hidden = []

		### symbolic theano variables ### 
		# Note that we are using itensor3 as we 3D one-hot-encoded input (integers)
		self.x_pitch_sym = T.itensor3('x_pitch_sym')
		self.x_duration_sym = T.itensor3('x_duration_sym')

		self.y_pitch_sym = T.itensor3('y_pitch_sym')
		self.y_duration_sym = T.itensor3('y_duration_sym')

		self.mask_sym = T.matrix('mask_sym')

		##### THE LAYERS OF THE NEXT-STEP PREDICTION GRU NETWORK #####

		### INPUT NETWORK ###
		# Two input layers receiving Onehot-encoded data
		l_in_pitch = InputLayer((None, None, self.num_features_pitch), name="l_in_pitch")
		l_in_duration = InputLayer((None, None, self.num_features_duration), name="l_in_duration")

		# Layer merging the two input layers
		l_in_merge = ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")
		# The mask layer for ignoring time-steps after <eos> in the GRU layer
		l_in_mask = InputLayer((None, self.max_seq_len), name="l_in_mask")


		### OUTPUT NETWORK ###
		# Simple input layer that the GRU layer can feed it's hidden states to
		l_out_in = InputLayer((None, self.num_gru_layer_units), name="l_out_in")

		# Two dense layers with softmax output (prediction probabilities)
		l_out_softmax_pitch = DenseLayer(l_out_in, num_units=self.num_features_pitch, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_pitch')
		l_out_softmax_duration = DenseLayer(l_out_in, num_units=self.num_features_duration, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_duration')

		# A layer for merging the two one-hot-encoding layers, so the GRU layer can get this as output_network and feed the previous output into the next prediction step.
		l_out_merge = ConcatLayer([l_out_softmax_pitch, l_out_softmax_duration], axis=-1, name="l_out_merge")

		### GRU LAYER ###
		# Main part of the model: 
		# The Gated-Recurrent-Unit (GRU) layer receiving both the original target at time t and the networks previous onehot-output from time t-1
		self.l_out_gru = GRUOutputInLayer(l_in_merge, l_out_merge, num_units=self.num_gru_layer_units, name='GRULayer', mask_input=l_in_mask, use_onehot_previous_output=self.use_deterministic_previous_output)

		# Setting up the output-layers as softmax-encoded pitch and duration vectors from the dense layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		self.l_out_pitch = SliceLayer(self.l_out_gru, indices=slice(self.num_features_pitch), axis=-1, name="SliceLayer_pitch")

		self.l_out_duration = SliceLayer(self.l_out_gru, indices=slice(self.num_features_pitch, self.num_features_pitch + self.num_features_duration), axis=-1, name="SliceLayer_duration")

		### NETWORK OUTPUTS ###
		# Setting up the output as softmax-encoded pitch and duration vectors from the dense softmax layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		output_pitch = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)
		output_duration = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)

		# Compute costs
		cost_pitch, acc_pitch = eval(output_pitch, self.y_pitch_sym, self.num_features_pitch, self.mask_sym)
		cost_duration, acc_duration = eval(output_duration, self.y_duration_sym, self.num_features_duration, self.mask_sym)
		total_cost = cost_pitch + cost_duration

		#Get parameters of both encoder and decoder
		all_parameters = get_all_params([self.l_out_pitch, self.l_out_duration], trainable=True)

		print "Trainable Model Parameters"
		print "-"*40
		for param in all_parameters:
		    print param, param.get_value().shape
		print "-"*40

		#add grad clipping to avoid exploding gradients
		all_grads = [T.clip(g,-3,3) for g in T.grad(total_cost, all_parameters)]
		all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

		#Compile Theano functions.
		updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

		self.f_train = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration], updates=updates)
		#since we don't have any stochasticity in the network we will just use the training graph without any updates given
		self.f_eval = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration])


class GRU_Network(MusicModelGRU):
	"""docstring for GRU_Network"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25):
		#super(GRU_Network, self).__init__()
		
		# Model naming and data path
		self.model_name = model_name
		self.model_data_path = "../data/models/"

		# Model hyperparameters
		self.num_gru_layer_units = num_gru_layer_units
		self.use_deterministic_previous_output = False
		
		# Model feature dimensions
		self.max_seq_len = max_seq_len-1
		self.num_features_pitch = num_features_pitch
		self.num_features_duration = num_features_duration
		
		# Training metadata
		self.batch_size = 10
		self.number_of_epochs_trained = 0

		### Learning curve data ### 
		# Categorical Crossentropy Costs 
		self.cost_train_pitch = []
		self.cost_train_duration = []
		self.cost_valid_pitch = []
		self.cost_valid_duration = []
		
		# Accuracies
		self.acc_train_pitch = []
		self.acc_train_duration = []
		self.acc_valid_pitch = []
		self.acc_valid_duration = []

		# Norms over horizontal GRU weights
		self.horz_update = []
		self.horz_reset = []
		self.horz_hidden = []

		# Norms over vertical GRU weights
		self.vert_update = []
		self.vert_reset = []
		self.vert_hidden = []

		### symbolic theano variables ### 
		# Note that we are using itensor3 as we 3D one-hot-encoded input (integers)
		self.x_pitch_sym = T.itensor3('x_pitch_sym')
		self.x_duration_sym = T.itensor3('x_duration_sym')

		self.y_pitch_sym = T.itensor3('y_pitch_sym')
		self.y_duration_sym = T.itensor3('y_duration_sym')

		self.mask_sym = T.matrix('mask_sym')



		##### THE LAYERS OF THE NEXT-STEP PREDICTION GRU NETWORK #####

		### INPUT NETWORK ###
		# Two input layers receiving Onehot-encoded data
		l_in_pitch = InputLayer((None, None, self.num_features_pitch), name="l_in_pitch")
		l_in_duration = InputLayer((None, None, self.num_features_duration), name="l_in_duration")

		# Layer merging the two input layers
		l_in_merge = ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")
		# The mask layer for ignoring time-steps after <eos> in the GRU layer
		l_in_mask = InputLayer((None, self.max_seq_len), name="l_in_mask")


		### OUTPUT NETWORK ###
		# Simple input layer that the GRU layer can feed it's hidden states to
		self.l_out_gru = GRULayer(l_in_merge, num_units=self.num_gru_layer_units, name='GRULayer', mask_input=l_in_mask)

		# We need to do some reshape voodo to connect a softmax layer to the decoder.
		# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples 
		# In short this line changes the shape from 
		# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units). 
		# We need to do this since the softmax is applied to the last dimension and we want to 
		# softmax the output at each position individually
		l_out_reshape = ReshapeLayer(self.l_out_gru, (-1, [2]), name="l_out_reshape")


		# Setting up the output-layers as softmax-encoded pitch and duration vectors from the dense layers. (Two dense layers with softmax output, e.g. prediction probabilities for next note in melody)
		l_out_softmax_pitch = DenseLayer(l_out_reshape, num_units=self.num_features_pitch, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_pitch')
		l_out_softmax_duration = DenseLayer(l_out_reshape, num_units=self.num_features_duration, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_duration')

		# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing 
		#us to use different batch sizes in the model.
		self.l_out_pitch = ReshapeLayer(l_out_softmax_pitch, (-1, self.max_seq_len, self.num_features_pitch), name="l_out_pitch")	
		self.l_out_duration = ReshapeLayer(l_out_softmax_duration, (-1, self.max_seq_len, self.num_features_duration), name="l_out_duration")

		### NETWORK OUTPUTS ###
		# Setting up the output as softmax-encoded pitch and duration vectors from the dense softmax layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		output_pitch = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)
		output_duration = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)

		# Compute costs
		cost_pitch, acc_pitch = eval(output_pitch, self.y_pitch_sym, self.num_features_pitch, self.mask_sym)
		cost_duration, acc_duration = eval(output_duration, self.y_duration_sym, self.num_features_duration, self.mask_sym)
		total_cost = cost_pitch + cost_duration

		#Get parameters of both encoder and decoder
		all_parameters = get_all_params([self.l_out_pitch, self.l_out_duration], trainable=True)

		print "Trainable Model Parameters"
		print "-"*40
		for param in all_parameters:
		    print param, param.get_value().shape
		print "-"*40

		#add grad clipping to avoid exploding gradients
		all_grads = [T.clip(g,-3,3) for g in T.grad(total_cost, all_parameters)]
		all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

		#Compile Theano functions.
		updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

		self.f_train = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration], updates=updates)
		#since we don't have any stochasticity in the network we will just use the training graph without any updates given
		self.f_eval = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration])


class GRU_Network_Continuous(MusicModelGRU):
	"""docstring for GRU_Network"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25):
		#super(GRU_Network, self).__init__()
		
		# Model naming and data path
		self.model_name = model_name
		self.model_data_path = "../data/models/"

		# Model hyperparameters
		self.num_gru_layer_units = num_gru_layer_units
		self.use_deterministic_previous_output = False
		
		# Model feature dimensions
		self.max_seq_len = max_seq_len-1
		self.num_features_pitch = num_features_pitch
		self.num_features_duration = num_features_duration
		
		# Training metadata
		self.batch_size = 10
		self.number_of_epochs_trained = 0

		### Learning curve data ### 
		# Categorical Crossentropy Costs 
		self.cost_train_pitch = []
		self.cost_train_duration = []
		self.cost_valid_pitch = []
		self.cost_valid_duration = []
		
		# Accuracies
		self.acc_train_pitch = []
		self.acc_train_duration = []
		self.acc_valid_pitch = []
		self.acc_valid_duration = []

		# Norms over horizontal GRU weights
		self.horz_update = []
		self.horz_reset = []
		self.horz_hidden = []

		# Norms over vertical GRU weights
		self.vert_update = []
		self.vert_reset = []
		self.vert_hidden = []

		### symbolic theano variables ### 
		# Note that we are using itensor3 as we 3D one-hot-encoded input (integers)
		self.x_pitch_sym = T.itensor3('x_pitch_sym')
		self.x_duration_sym = T.itensor3('x_duration_sym')

		self.y_pitch_sym = T.itensor3('y_pitch_sym')
		self.y_duration_sym = T.itensor3('y_duration_sym')

		self.mask_sym = T.matrix('mask_sym')



		##### THE LAYERS OF THE NEXT-STEP PREDICTION GRU NETWORK #####

		### INPUT NETWORK ###
		# Two input layers receiving Onehot-encoded data
		l_in_pitch = InputLayer((None, None, self.num_features_pitch), name="l_in_pitch")
		l_in_duration = InputLayer((None, None, self.num_features_duration), name="l_in_duration")

		# Layer merging the two input layers
		l_in_merge = ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")
		# The mask layer for ignoring time-steps after <eos> in the GRU layer
		l_in_mask = InputLayer((None, self.max_seq_len), name="l_in_mask")


		### OUTPUT NETWORK ###
		# Simple input layer that the GRU layer can feed it's hidden states to
		self.l_out_gru = GRULayer(l_in_merge, num_units=self.num_gru_layer_units, name='GRULayer', mask_input=l_in_mask)

		# We need to do some reshape voodo to connect a softmax layer to the decoder.
		# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples 
		# In short this line changes the shape from 
		# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units). 
		# We need to do this since the softmax is applied to the last dimension and we want to 
		# softmax the output at each position individually
		l_out_reshape = ReshapeLayer(self.l_out_gru, (-1, [2]), name="l_out_reshape")


		# Setting up the output-layers as softmax-encoded pitch and duration vectors from the dense layers. (Two dense layers with softmax output, e.g. prediction probabilities for next note in melody)
		l_out_softmax_pitch = DenseLayer(l_out_reshape, num_units=self.num_features_pitch, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_pitch')
		l_out_softmax_duration = DenseLayer(l_out_reshape, num_units=self.num_features_duration, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_duration')

		# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing 
		#us to use different batch sizes in the model.
		self.l_out_pitch = ReshapeLayer(l_out_softmax_pitch, (-1, self.max_seq_len, self.num_features_pitch), name="l_out_pitch")	
		self.l_out_duration = ReshapeLayer(l_out_softmax_duration, (-1, self.max_seq_len, self.num_features_duration), name="l_out_duration")

		### NETWORK OUTPUTS ###
		# Setting up the output as softmax-encoded pitch and duration vectors from the dense softmax layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		output_pitch = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)
		output_duration = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.mask_sym}, deterministic = False)

		# Compute costs
		cost_pitch, acc_pitch = eval(output_pitch, self.y_pitch_sym, self.num_features_pitch, self.mask_sym)
		cost_duration, acc_duration = eval(output_duration, self.y_duration_sym, self.num_features_duration, self.mask_sym)
		total_cost = cost_pitch + cost_duration

		#Get parameters of both encoder and decoder
		all_parameters = get_all_params([self.l_out_pitch, self.l_out_duration], trainable=True)

		print "Trainable Model Parameters"
		print "-"*40
		for param in all_parameters:
		    print param, param.get_value().shape
		print "-"*40

		#add grad clipping to avoid exploding gradients
		all_grads = [T.clip(g,-3,3) for g in T.grad(total_cost, all_parameters)]
		all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

		#Compile Theano functions.
		updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

		self.f_train = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration], updates=updates)
		#since we don't have any stochasticity in the network we will just use the training graph without any updates given
		self.f_eval = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration])

def eval(output, target, num_features, mask):
	### Evalutation function returning cost and accuracy given predictions
	output_reshaped = T.reshape(output, (-1, num_features))
	target_reshaped = T.reshape(target, (-1, num_features))

	#cost function
	total_cost = T.nnet.categorical_crossentropy(output_reshaped
	    , target_reshaped)
	flat_mask = mask.flatten(ndim=1)
	N = T.dot(flat_mask, flat_mask)
	cost = T.dot(flat_mask, total_cost) / N

	#accuracy function
	is_equal = T.eq(T.argmax(output_reshaped,axis=-1),T.argmax(target_reshaped,axis=-1))
	acc = T.dot(flat_mask, is_equal) / N  # gives float64 because eq is uint8, T.cast(eq, 'float32') will fix that...
	return cost, acc

def data_setup(data):
	# Receive: 
	# original data: x = {"pitch": x_pitch, "duration: x_duration, "mask": x_mask] (a list of 3 numpy arrays) with dimensions:
		# dim(x_pitch) = (N, max_seq_len, num_features_pitch)
		# dim(x_duration) = (N, max_seq_len, num_features_duration)
		# dim(x_mask) = (N, max_seq_len)
	# And reform the original data into: 
		# input: x_inpu = [x_{0}, ..., x_{max_seq_len-1}] with dim(x)=(N, max_seq_len-1, num_features)
		# target: y = [x_{1}, ..., x_{max_seq_len}] with dim(y)=(N, max_seq_len-1, num_features) - this is one-step-ahead target that the model will predict.

	# Inputs
	x_pitch = data["pitch"][:,:-1]
	x_duration = data["duration"][:,:-1]

	# Masks (meta-input which is ones until end-of-sequence (melody) and zeros for the rest which the model does not use)
	mask = data["mask"][:,:-1]

	# Targets
	y_pitch = data["pitch"][:,1:]
	y_duration = data["duration"][:,1:]

	return x_pitch, y_pitch, x_duration, y_duration, mask