from data import load_data, array2midi, one_hot_decoder, plotActivations
# from aux import _path
from grulayer import GRUOutputInLayer
import cPickle as pickle
import os.path
from os import listdir

import lasagne
from lasagne.layers import (
    InputLayer, DenseLayer, GRULayer, ConcatLayer, SliceLayer, ReshapeLayer, DropoutLayer,
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
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25, set_x_input_to_zero=False):
		#super(MusicModelGRU, self).__init__()
		
		# Model naming and data path
		self.model_name = model_name
		self.model_data_path = "../data/models/"

		# Model hyperparameters
		self.num_gru_layer_units = num_gru_layer_units
		self.set_x_input_to_zero = set_x_input_to_zero
		self.use_deterministic_previous_output = True


		# Model feature dimensions
		self.max_seq_len = max_seq_len-1
		self.num_features_pitch = num_features_pitch
		self.num_features_duration = num_features_duration
		
		# Training metadata
		self.batch_size = 10
		self.number_of_epochs_trained = 0


		self.eps = 1e-6
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

		# frobenius norms over horizontal GRU weights
		self.horz_update_norm = []
		self.horz_reset_norm = []
		self.horz_hidden_norm = []

		# frobenius norms over vertical GRU weights
		self.vert_update_norm = []
		self.vert_reset_norm = []
		self.vert_hidden_norm = []

		# mean of horizontal GRU weights
		self.horz_update_mean = []
		self.horz_reset_mean = []
		self.horz_hidden_mean = []

		# mean of vertical GRU weights
		self.vert_update_mean = []
		self.vert_reset_mean = []
		self.vert_hidden_mean = []

		# positive value fraction of horizontal GRU weights
		self.horz_update_pos = []
		self.horz_reset_pos = []
		self.horz_hidden_pos = []

		# positive value fraction of vertical GRU weights
		self.vert_update_pos = []
		self.vert_reset_pos = []
		self.vert_hidden_pos = []

		# negative value fraction of horizontal GRU weights
		self.horz_update_neg = []
		self.horz_reset_neg = []
		self.horz_hidden_neg = []

		# negative value fraction of vertical GRU weights
		self.vert_update_neg = []
		self.vert_reset_neg = []
		self.vert_hidden_neg = []

		### symbolic theano variables ### 
		# Note that we are using itensor3 as we 3D one-hot-encoded input (integers)
		self.x_pitch_sym = T.itensor3('x_pitch_sym')
		self.x_duration_sym = T.itensor3('x_duration_sym')

		self.y_pitch_sym = T.itensor3('y_pitch_sym')
		self.y_duration_sym = T.itensor3('y_duration_sym')

		self.x_mask_sym = T.matrix('x_mask_sym')
		self.y_mask_sym = T.matrix('y_mask_sym')
		

	
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
		x_pitch_train, y_pitch_train, x_duration_train, y_duration_train, x_mask_train, y_mask_train = data_setup(train_data, self.set_x_input_to_zero)	

		# Validation data
		if valid_data is not None:
			x_pitch_valid, y_pitch_valid, x_duration_valid, y_duration_valid, x_mask_valid, y_mask_valid = data_setup(valid_data, self.set_x_input_to_zero)	

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
				x_mask_batch = x_mask_train[subset]
				y_mask_batch = y_mask_train[subset]
				# Train for batch and collect cost, accuracy and output
				self.f_train(x_pitch_batch, y_pitch_batch, x_duration_batch, y_duration_batch, x_mask_batch, y_mask_batch)
				# epoch_cost += batch_cost
			train_cost_pitch, train_acc_pitch, train_output_pitch, train_cost_duration, train_acc_duration, train_output_duration = self.f_eval(x_pitch_train, y_pitch_train, x_duration_train, y_duration_train, x_mask_train, y_mask_train)
			train_string = "Train: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(train_cost_pitch), float(train_cost_duration), float(train_acc_pitch), float(train_acc_duration))

			self.update_weight_measures()

			self.cost_train_pitch += [train_cost_pitch]
			self.acc_train_pitch += [train_acc_pitch]
			self.cost_train_duration += [train_cost_duration]
			self.acc_train_duration += [train_acc_duration]

			if valid_data is not None:
				valid_cost_pitch, valid_acc_pitch, valid_output_pitch, valid_cost_duration, valid_acc_duration, valid_output_duration = self.f_eval(x_pitch_valid, y_pitch_valid, x_duration_valid, y_duration_valid, x_mask_valid, y_mask_valid)
				self.cost_valid_pitch += [valid_cost_pitch]
				self.acc_valid_pitch += [valid_acc_pitch]
				self.cost_valid_duration += [valid_cost_duration]
				self.acc_valid_duration += [valid_acc_duration]
				valid_string = "Valid: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(valid_cost_pitch), float(valid_cost_duration), float(valid_acc_pitch), float(valid_acc_duration))
			
			epoch_string = "\nEpoch {:2d}: {}\n{}\n{}".format(epoch + 1, header_string, train_string, valid_string)
			print(epoch_string)

		# Update the number of epochs the model have been trained for
		self.number_of_epochs_trained += num_epochs


	def update_weight_measures(self):
		# Compute frobenius norms over horizontal GRU weights
		self.horz_update_norm += [np.linalg.norm(self.l_out_gru.W_hid_to_updategate.get_value())]
		self.horz_reset_norm += [np.linalg.norm(self.l_out_gru.W_hid_to_resetgate.get_value())]
		self.horz_hidden_norm += [np.linalg.norm(self.l_out_gru.W_hid_to_hidden_update.get_value())]

		# Compute frobenius norms over vertical GRU weights
		self.vert_update_norm += [np.linalg.norm(self.l_out_gru.W_in_to_updategate.get_value())]
		self.vert_reset_norm += [np.linalg.norm(self.l_out_gru.W_in_to_resetgate.get_value())]
		self.vert_hidden_norm += [np.linalg.norm(self.l_out_gru.W_in_to_hidden_update.get_value())]

		# Compute mean of horizontal GRU weights
		self.horz_update_mean += [self.l_out_gru.W_hid_to_updategate.get_value().mean()]
		self.horz_reset_mean += [self.l_out_gru.W_hid_to_resetgate.get_value().mean()]
		self.horz_hidden_mean += [self.l_out_gru.W_hid_to_hidden_update.get_value().mean()]

		# Compute mean of vertical GRU weights
		self.vert_update_mean += [self.l_out_gru.W_in_to_updategate.get_value().mean()]
		self.vert_reset_mean += [self.l_out_gru.W_in_to_resetgate.get_value().mean()]
		self.vert_hidden_mean += [self.l_out_gru.W_in_to_hidden_update.get_value().mean()]

		# Compute positive value fraction of horizontal GRU weights
		self.horz_update_pos += [(self.l_out_gru.W_hid_to_updategate.get_value() > self.eps).mean()]
		self.horz_reset_pos += [(self.l_out_gru.W_hid_to_resetgate.get_value() > self.eps).mean()]
		self.horz_hidden_pos += [(self.l_out_gru.W_hid_to_hidden_update.get_value() > self.eps).mean()]

		# Compute positive value fraction of vertical GRU weights
		self.vert_update_pos += [(self.l_out_gru.W_in_to_updategate.get_value() > self.eps).mean()]
		self.vert_reset_pos += [(self.l_out_gru.W_in_to_resetgate.get_value() > self.eps).mean()]
		self.vert_hidden_pos += [(self.l_out_gru.W_in_to_hidden_update.get_value() > self.eps).mean()]

		# Compute negative value fraction of horizontal GRU weights
		self.horz_update_neg += [(self.l_out_gru.W_hid_to_updategate.get_value() < -self.eps).mean()]
		self.horz_reset_neg += [(self.l_out_gru.W_hid_to_resetgate.get_value() < -self.eps).mean()]
		self.horz_hidden_neg += [(self.l_out_gru.W_hid_to_hidden_update.get_value() < -self.eps).mean()]

		# Compute negative value fraction of vertical GRU weights
		self.vert_update_neg += [(self.l_out_gru.W_in_to_updategate.get_value() < -self.eps).mean()]
		self.vert_reset_neg += [(self.l_out_gru.W_in_to_resetgate.get_value() < -self.eps).mean()]
		self.vert_hidden_neg += [(self.l_out_gru.W_in_to_hidden_update.get_value() < -self.eps).mean()]

	def evaluate(self, data, dropout_range=None, dropout_fraction=None, write2midi=False, pitch_map=None, duration_map=None, plot_GRU_activations=False):
		# Model reconstructions on the given evaluation data:
			# original data: x = {"pitch": x_pitch, "duration: x_duration, "mask": x_mask] (a list of 3 numpy arrays) with dimensions:
				# dim(x_pitch) = (N, max_seq_len, num_features_pitch)
				# dim(x_duration) = (N, max_seq_len, num_features_duration)
				# dim(x_mask) = (N, max_seq_len)
			# The original data is reformed into: 
				# input: x_inpu = [x_{0}, ..., x_{max_seq_len-1}] with dim(x)=(N, max_seq_len-1, num_features)
				# target: y = [x_{1}, ..., x_{max_seq_len}] with dim(y)=(N, max_seq_len-1, num_features) - this is one-step-ahead target that the model will predict.
		x_pitch, y_pitch, x_duration, y_duration, x_mask, y_mask = data_setup(data, self.set_x_input_to_zero)	
		metadata = data["metadata"]
		indices = data["indices"]
		N = x_pitch.shape[0]
		filename = self.model_data_path + self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained)

		if plot_GRU_activations:
			output_gru = self.f_eval_gru(x_pitch, x_duration, x_mask)
			plotActivations(filename, output_gru, x_mask, x_pitch, pitch_map, x_duration, duration_map, metadata, indices)


		if dropout_range is not None:
			zero_array_pitch = np.zeros((len(dropout_range), self.num_features_pitch))
			zero_array_duration = np.zeros((len(dropout_range), self.num_features_duration))
			for i in range(N):
				x_pitch[i, dropout_range] = zero_array_pitch
				x_duration[i, dropout_range] = zero_array_duration

		if dropout_fraction is not None:
			fraction_stop = int(dropout_fraction*self.max_seq_len)
			zero_array_pitch = np.zeros((fraction_stop, self.num_features_pitch))
			zero_array_duration = np.zeros((fraction_stop, self.num_features_duration))
			for i in range(N):
				fraction_indices = np.random.permutation(self.max_seq_len)[:fraction_stop]
				x_pitch[i, fraction_indices] = zero_array_pitch
				x_duration[i, fraction_indices] = zero_array_duration				


		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = self.f_eval(x_pitch, y_pitch, x_duration, y_duration, x_mask, y_mask)



		if write2midi:
			filename = self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained)
			if dropout_range is not None:
				filename += "_dropRange_({},{})".format(np.min(dropout_range), np.max(dropout_range))
			if dropout_fraction is not None:
				filename += "_dropFraction_{:.2f}".format(dropout_fraction)
			metadata = data["metadata"]
			indices = data["indices"]

			array2midi(data["pitch"], pitch_map, data["duration"], duration_map, metadata, indices, filepath=self.model_data_path, filename=filename + "_original")
			array2midi(output_pitch, pitch_map, output_duration, duration_map, metadata, indices, filepath=self.model_data_path, filename=filename + "_reconstruction")

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

		# frobenius norms over horizontal GRU weights
		model["horz_update_norm"] = self.horz_update_norm
		model["horz_reset_norm"] = self.horz_reset_norm
		model["horz_hidden_norm"] = self.horz_hidden_norm

		# frobenius norms over vertical GRU weights
		model["vert_update_norm"] = self.vert_update_norm
		model["vert_reset_norm"] = self.vert_reset_norm
		model["vert_hidden_norm"] = self.vert_hidden_norm

		# mean of horizontal GRU weights
		model["horz_update"] = self.horz_update_mean
		model["horz_reset"] = self.horz_reset_mean
		model["horz_hidden"] = self.horz_hidden_mean

		# mean of vertical GRU weights
		model["vert_update"] = self.vert_update_mean
		model["vert_reset"] = self.vert_reset_mean
		model["vert_hidden"] = self.vert_hidden_mean

		# positive value fraction of horizontal GRU weights
		model["horz_update_pos"] = self.horz_update_pos
		model["horz_reset_pos"] = self.horz_reset_pos
		model["horz_hidden_pos"] = self.horz_hidden_pos

		# positive value fraction of vertical GRU weights
		model["vert_update_pos"] = self.vert_update_pos
		model["vert_reset_pos"] = self.vert_reset_pos
		model["vert_hidden_pos"] = self.vert_hidden_pos

		# negative value fraction of horizontal GRU weights
		model["horz_update_neg"] = self.horz_update_neg
		model["horz_reset_neg"] = self.horz_reset_neg
		model["horz_hidden_neg"] = self.horz_hidden_neg

		# negative value fraction of vertical GRU weights
		model["vert_update_neg"] = self.vert_update_neg
		model["vert_reset_neg"] = self.vert_reset_neg
		model["vert_hidden_neg"] = self.vert_hidden_neg

		with open(model_path, "wb") as file:
			pickle.dump(model, file)

	def load(self, number_of_epochs_trained=None):
		model_loaded = False

		### LOAD model ###
		model_name_spec = self.model_name + "_gru_{}_bs_{}_e_".format(self.num_gru_layer_units, self.batch_size)

		if number_of_epochs_trained is None:
			model_epochs = [int(file.split(".")[0].split("_")[-1]) for file in listdir(self.model_data_path) if (file[0] != "." and file[:len(model_name_spec)] == model_name_spec and file.split(".")[-1] == "pkl")]
		else: 
			model_epochs = [number_of_epochs_trained]

		# Check for latest or the number_of_epochs_trained model data
		if model_epochs:
			max_epoch_num = max(model_epochs)
			print("The current number of epochs the {} model have been trained is: {}\n".format(self.model_name, max_epoch_num))
			print("Loading the data for the current state of the model.\n")
			model_path = self.model_data_path + model_name_spec + str(max_epoch_num) + ".pkl"
			if os.path.isfile(model_path):
				with open(model_path, "rb") as file:
					model = pickle.load(file)
				model_loaded = True
				print("Loaded {}\n".format(model_path))
		else: 
			print("No previous data on this model exists. Use the methods train() and save() first and then load().\n")

		if model_loaded:
			print("Setting up model with previous parameters from the file {}\n".format(model_path))

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

			# frobenius norms over horizontal GRU weights
			self.horz_update_norm = model["horz_update_norm"]
			self.horz_reset_norm = model["horz_reset_norm"]
			self.horz_hidden_norm = model["horz_hidden_norm"]

			# frobenius norms over vertical GRU weights
			self.vert_update_norm = model["vert_update_norm"]
			self.vert_reset_norm = model["vert_reset_norm"]
			self.vert_hidden_norm = model["vert_hidden_norm"]

			# mean of horizontal GRU weights
			self.horz_update_mean = model["horz_update"]
			self.horz_reset_mean = model["horz_reset"]
			self.horz_hidden_mean = model["horz_hidden"]

			# mean of vertical GRU weights
			self.vert_update_mean = model["vert_update"]
			self.vert_reset_mean = model["vert_reset"]
			self.vert_hidden_mean = model["vert_hidden"]

			# positive value fraction of horizontal GRU weights
			self.horz_update_pos = model["horz_update_pos"]
			self.horz_reset_pos = model["horz_reset_pos"]
			self.horz_hidden_pos = model["horz_hidden_pos"]

			# positive value fraction of vertical GRU weights
			self.vert_update_pos = model["vert_update_pos"]
			self.vert_reset_pos = model["vert_reset_pos"]
			self.vert_hidden_pos = model["vert_hidden_pos"]

			# negative value fraction of horizontal GRU weights
			self.horz_update_neg = model["horz_update_neg"]
			self.horz_reset_neg = model["horz_reset_neg"]
			self.horz_hidden_neg = model["horz_hidden_neg"]

			# negative value fraction of vertical GRU weights
			self.vert_update_neg = model["vert_update_neg"]
			self.vert_reset_neg = model["vert_reset_neg"]
			self.vert_hidden_neg = model["vert_hidden_neg"]

	def plotLearningCurves(self):
		model_path = self.model_data_path + self.model_name + "_gru_{}_bs_{}_e_{}".format(self.num_gru_layer_units, self.batch_size, self.number_of_epochs_trained)
		epochs = range(1, self.number_of_epochs_trained+1)

		# Accuracy plots
		plt.figure()
		acc_train_pitch_plt, = plt.plot(epochs, self.acc_train_pitch, 'r-')
		acc_valid_pitch_plt, = plt.plot(epochs, self.acc_valid_pitch, 'r--')
		acc_train_duration_plt, = plt.plot(epochs, self.acc_train_duration, 'b-')
		acc_valid_duration_plt, = plt.plot(epochs, self.acc_valid_duration, 'b--')
		plt.ylabel('Accuracies')
		plt.xlabel('Epoch #')
		plt.legend([acc_train_pitch_plt, acc_valid_pitch_plt, acc_train_duration_plt, acc_valid_duration_plt], ["Training Pitch", "Validation Pitch", "Training Duration", "Validation Duration"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_acc.png")

		# Cost plots
		plt.figure()
		cost_train_pitch_plt, = plt.plot(epochs, self.cost_train_pitch, 'r-')
		cost_valid_pitch_plt, = plt.plot(epochs, self.cost_valid_pitch, 'r--')
		cost_train_duration_plt, = plt.plot(epochs, self.cost_train_duration, 'b-')
		cost_valid_duration_plt, = plt.plot(epochs, self.cost_valid_duration, 'b--')
		plt.ylabel('Crossentropy Costs')
		plt.xlabel('Epoch #')
		plt.legend([cost_train_pitch_plt, cost_valid_pitch_plt, cost_train_duration_plt, cost_valid_duration_plt], ["Training Pitch", "Validation Pitch", "Training Duration", "Validation Duration"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_cost.png")
		
		plt.close("all")

		# Horizontal weights plots
		plt.figure()
		horz_update_plt, = plt.plot(epochs, self.horz_update_mean)
		horz_reset_plt, = plt.plot(epochs, self.horz_reset_mean)
		horz_hidden_plt, = plt.plot(epochs, self.horz_hidden_mean)
		plt.ylabel('Mean of Horizontal Weights')
		plt.xlabel('Epoch #')
		plt.legend([horz_update_plt, horz_reset_plt, horz_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_horzWeights_mean.png")

		# Vertical weights plots
		plt.figure()
		vert_update_plt, = plt.plot(epochs, self.vert_update_mean)
		vert_reset_plt, = plt.plot(epochs, self.vert_reset_mean)
		vert_hidden_plt, = plt.plot(epochs, self.vert_hidden_mean)
		plt.ylabel('Mean of Vertical Weights')
		plt.xlabel('Epoch #')
		plt.legend([vert_update_plt, vert_reset_plt, vert_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_vertWeights_mean.png")

		plt.close("all")

		# Horizontal weights plots
		plt.figure()
		horz_update_plt, = plt.plot(epochs, self.horz_update_norm)
		horz_reset_plt, = plt.plot(epochs, self.horz_reset_norm)
		horz_hidden_plt, = plt.plot(epochs, self.horz_hidden_norm)
		plt.ylabel('Frobenius norm of Horizontal Weights')
		plt.xlabel('Epoch #')
		plt.legend([horz_update_plt, horz_reset_plt, horz_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_horzWeights_norm.png")

		# Vertical weights plots
		plt.figure()
		vert_update_plt, = plt.plot(epochs, self.vert_update_norm)
		vert_reset_plt, = plt.plot(epochs, self.vert_reset_norm)
		vert_hidden_plt, = plt.plot(epochs, self.vert_hidden_norm)
		plt.ylabel('Frobenius norm of Vertical Weights')
		plt.xlabel('Epoch #')
		plt.legend([vert_update_plt, vert_reset_plt, vert_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_vertWeights_norm.png")

		plt.close("all")

		# Horizontal weights plots
		plt.figure()
		horz_update_plt, = plt.plot(epochs, self.horz_update_pos)
		horz_reset_plt, = plt.plot(epochs, self.horz_reset_pos)
		horz_hidden_plt, = plt.plot(epochs, self.horz_hidden_pos)
		plt.ylabel('Fraction of Positive values in Horizontal Weights')
		plt.xlabel('Epoch #')
		plt.legend([horz_update_plt, horz_reset_plt, horz_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_horzWeights_pos.png")

		# Vertical weights plots
		plt.figure()
		vert_update_plt, = plt.plot(epochs, self.vert_update_pos)
		vert_reset_plt, = plt.plot(epochs, self.vert_reset_pos)
		vert_hidden_plt, = plt.plot(epochs, self.vert_hidden_pos)
		plt.ylabel('Fraction of Positive values in Vertical Weights')
		plt.xlabel('Epoch #')
		plt.legend([vert_update_plt, vert_reset_plt, vert_hidden_plt], ["Update Gate", "Reset Gate", "Candidate Gate"])
		#plt.title('', fontsize=20)
		plt.grid('on')
		plt.savefig(model_path + "_vertWeights_pos.png")

		plt.close("all")

class GRU_Network_Using_Previous_Output(MusicModelGRU):
	"""docstring for GRU_Network_Using_Previous_Output"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25, set_x_input_to_zero=False, use_deterministic_previous_output=True, in_dropout_p=0):
		super(GRU_Network_Using_Previous_Output, self).__init__(model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units, set_x_input_to_zero)
		
		self.use_deterministic_previous_output = use_deterministic_previous_output
		self.in_dropout_p = in_dropout_p
		##### THE LAYERS OF THE NEXT-STEP PREDICTION GRU NETWORK #####

		### INPUT NETWORK ###
		# Two input layers receiving Onehot-encoded data
		l_in_pitch = InputLayer((None, None, self.num_features_pitch), name="l_in_pitch")
		l_in_duration = InputLayer((None, None, self.num_features_duration), name="l_in_duration")

		# Layer merging the two input layers
		l_in_merge = ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")

		l_in_intermediate = l_in_merge

		if in_dropout_p > 0:
			l_in_intermediate = DropoutLayer(l_in_intermediate, rescale=False, p=in_dropout_p, shared_axes=tuple([1,2]))

		# The mask layer for ignoring time-steps after <eos> in the GRU layer
		l_in_mask = InputLayer((None, self.max_seq_len), name="l_in_mask")


		### OUTPUT NETWORK ###
		# Simple input layer that the GRU layer can feed it's hidden states to
		l_out_in = InputLayer((None, self.num_gru_layer_units), name="l_out_in")

		# l_out_intermediate = l_out_in
		# if out_dropout_p > 0:
		# 	l_out_intermediate = DropoutLayer(l_out_intermediate, rescale=False, p=out_dropout_p)


		# Two dense layers with softmax output (prediction probabilities)
		l_out_softmax_pitch = DenseLayer(l_out_in, num_units=self.num_features_pitch, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_pitch')
		l_out_softmax_duration = DenseLayer(l_out_in, num_units=self.num_features_duration, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_duration')

		# A layer for merging the two one-hot-encoding layers, so the GRU layer can get this as output_network and feed the previous output into the next prediction step.
		l_out_merge = ConcatLayer([l_out_softmax_pitch, l_out_softmax_duration], axis=-1, name="l_out_merge")

		### GRU LAYER ###
		# Main part of the model: 
		# The Gated-Recurrent-Unit (GRU) layer receiving both the original target at time t and the networks previous onehot-output from time t-1
		self.l_out_gru = GRUOutputInLayer(l_in_intermediate, l_out_merge, num_units=self.num_gru_layer_units, name='GRULayer', mask_input=l_in_mask, use_onehot_previous_output=self.use_deterministic_previous_output)

		# Setting up the output-layers as softmax-encoded pitch and duration vectors from the dense layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		self.l_out_pitch = SliceLayer(self.l_out_gru, indices=slice(self.num_features_pitch), axis=-1, name="SliceLayer_pitch")

		self.l_out_duration = SliceLayer(self.l_out_gru, indices=slice(self.num_features_pitch, self.num_features_pitch + self.num_features_duration), axis=-1, name="SliceLayer_duration")

		### NETWORK OUTPUTS ###
		# Setting up the output as softmax-encoded pitch and duration vectors from the dense softmax layers.
		# (OBS: This is bypassing the onehot layers, so we evaluate the model on the softmax-outputs directly)
		output_pitch_train = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = False)
		output_duration_train = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = False)

		output_pitch_eval = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)
		output_duration_eval = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)

		output_gru = get_output(self.l_out_gru, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)

		# Compute costs
		# For indeterministic training 
		cost_pitch_train, acc_pitch_train = eval(output_pitch_train, self.y_pitch_sym, self.num_features_pitch, self.y_mask_sym)
		cost_duration_train, acc_duration_train = eval(output_duration_train, self.y_duration_sym, self.num_features_duration, self.y_mask_sym)
		total_cost = cost_pitch_train + cost_duration_train

		# and deterministic evaluation
		cost_pitch_eval, acc_pitch_eval = eval(output_pitch_eval, self.y_pitch_sym, self.num_features_pitch, self.y_mask_sym)
		cost_duration_eval, acc_duration_eval = eval(output_duration_eval, self.y_duration_sym, self.num_features_duration, self.y_mask_sym)

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

		self.f_train = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.x_mask_sym, self.y_mask_sym], [cost_pitch_train, acc_pitch_train, output_pitch_train, cost_duration_train, acc_duration_train, output_duration_train], updates=updates)
		
		#since we have stochasticity in the network when dropout is used we will use the evaluation graph without any updates given and deterministic=True
		self.f_eval = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.x_mask_sym, self.y_mask_sym], [cost_pitch_eval, acc_pitch_eval, output_pitch_eval, cost_duration_eval, acc_duration_eval, output_duration_eval])

		self.f_eval_gru = theano.function([self.x_pitch_sym, self.x_duration_sym, self.x_mask_sym], output_gru)


class GRU_Network(MusicModelGRU):
	"""docstring for GRU_Network"""
	def __init__(self, model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units=25, set_x_input_to_zero=False, in_dropout_p=0, out_dropout_p=0):
		super(GRU_Network, self).__init__(model_name, max_seq_len, num_features_pitch, num_features_duration, num_gru_layer_units, set_x_input_to_zero)
		
		self.in_dropout_p = in_dropout_p
		self.out_dropout_p = out_dropout_p

		##### THE LAYERS OF THE NEXT-STEP PREDICTION GRU NETWORK #####

		### INPUT NETWORK ###
		# Two input layers receiving Onehot-encoded data
		l_in_pitch = InputLayer((None, None, self.num_features_pitch), name="l_in_pitch")
		l_in_duration = InputLayer((None, None, self.num_features_duration), name="l_in_duration")

		# Layer merging the two input layers
		l_in_merge = ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")

		# Dropout in input network
		l_in_intermediate = l_in_merge
		if self.in_dropout_p > 0:
			l_in_intermediate = DropoutLayer(l_in_intermediate, rescale=False, p=self.in_dropout_p, shared_axes=(1,2))


		# The mask layer for ignoring time-steps after <eos> in the GRU layer
		l_in_mask = InputLayer((None, self.max_seq_len), name="l_in_mask")


		### OUTPUT NETWORK ###
		# Simple input layer that the GRU layer can feed it's hidden states to
		self.l_out_gru = GRULayer(l_in_intermediate, num_units=self.num_gru_layer_units, name='GRULayer', mask_input=l_in_mask)

		# Dropout in output network
		l_out_intermediate = self.l_out_gru
		if self.out_dropout_p > 0:
			l_out_intermediate = DropoutLayer(l_out_intermediate, rescale=False, p=self.out_dropout_p)

		# We need to do some reshape voodo to connect a softmax layer to the decoder.
		# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples 
		# In short this line changes the shape from 
		# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units). 
		# We need to do this since the softmax is applied to the last dimension and we want to 
		# softmax the output at each position individually
		l_out_reshape = ReshapeLayer(l_out_intermediate, (-1, [2]), name="l_out_reshape")


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
		output_pitch_train = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = False)
		output_duration_train = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = False)

		output_pitch_eval = get_output(self.l_out_pitch, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)
		output_duration_eval = get_output(self.l_out_duration, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)

		output_gru = get_output(self.l_out_gru, {l_in_pitch: self.x_pitch_sym, l_in_duration: self.x_duration_sym, l_in_mask: self.x_mask_sym}, deterministic = True)

		# Compute costs
		# For indeterministic training 
		cost_pitch_train, acc_pitch_train = eval(output_pitch_train, self.y_pitch_sym, self.num_features_pitch, self.y_mask_sym)
		cost_duration_train, acc_duration_train = eval(output_duration_train, self.y_duration_sym, self.num_features_duration, self.y_mask_sym)
		total_cost = cost_pitch_train + cost_duration_train

		# and deterministic evaluation
		cost_pitch_eval, acc_pitch_eval = eval(output_pitch_eval, self.y_pitch_sym, self.num_features_pitch, self.y_mask_sym)
		cost_duration_eval, acc_duration_eval = eval(output_duration_eval, self.y_duration_sym, self.num_features_duration, self.y_mask_sym)

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

		self.f_train = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.x_mask_sym, self.y_mask_sym], [cost_pitch_train, acc_pitch_train, output_pitch_train, cost_duration_train, acc_duration_train, output_duration_train], updates=updates)
		
		#since we have stochasticity in the network when dropout is used we will use the evaluation graph without any updates given and deterministic=True.
		self.f_eval = theano.function([self.x_pitch_sym, self.y_pitch_sym, self.x_duration_sym, self.y_duration_sym, self.x_mask_sym, self.y_mask_sym], [cost_pitch_eval, acc_pitch_eval, output_pitch_eval, cost_duration_eval, acc_duration_eval, output_duration_eval])

		self.f_eval_gru = theano.function([self.x_pitch_sym, self.x_duration_sym, self.x_mask_sym], output_gru)


####### Helper functions ##########

def eval(output, target, num_features, mask):
	### Evalutation function returning cost and accuracy given predictions
	output_reshaped = T.reshape(output, (-1, num_features))
	target_reshaped = T.reshape(target, (-1, num_features))
	flat_mask = mask.flatten(ndim=1)
	#cost function
	total_cost = T.nnet.categorical_crossentropy(output_reshaped
	    , target_reshaped)
	#total_cost[T.isnan(total_cost)] = 0
	N = T.sum(flat_mask)# T.dot(flat_mask, flat_mask)
	cost = T.dot(flat_mask, total_cost) / N

	#accuracy function
	is_equal = T.eq(T.argmax(output_reshaped,axis=-1),T.argmax(target_reshaped,axis=-1))
	acc = T.dot(flat_mask, is_equal) / N  # gives float64 because eq is uint8, T.cast(eq, 'float32') will fix that...
	return cost, acc

def data_setup(data, set_x_input_to_zero=False):
	# Receive: 
	# original data: x = {"pitch": x_pitch, "duration: x_duration, "mask": x_mask] (a list of 3 numpy arrays) with dimensions:
		# dim(x_pitch) = (N, max_seq_len, num_features_pitch)
		# dim(x_duration) = (N, max_seq_len, num_features_duration)
		# dim(x_mask) = (N, max_seq_len)
	# And reform the original data into: 
		# input: x_inpu = [x_{0}, ..., x_{max_seq_len-1}] with dim(x)=(N, max_seq_len-1, num_features)
		# target: y = [x_{1}, ..., x_{max_seq_len}] with dim(y)=(N, max_seq_len-1, num_features) - this is one-step-ahead target that the model will predict.

	x_pitch_temp = data["pitch"][:,:-1]
	x_duration_temp = data["duration"][:,:-1]

	# Inputs
	if set_x_input_to_zero:
		x_pitch = np.zeros(x_pitch_temp.shape).astype('int32')
		x_duration = np.zeros(x_duration_temp.shape).astype('int32')
		x_pitch[:,0,:] = x_pitch_temp[:,0,:]
		x_duration[:,0,:] = x_duration_temp[:,0,:]
	else: 
		x_pitch = x_pitch_temp
		x_duration = x_duration_temp

	# Masks (meta-input which is ones until end-of-sequence (melody) and zeros for the rest which the model does not use)
	x_mask = data["mask"][:,:-1]
	y_mask = data["mask"][:,1:]

	# Targets
	y_pitch = data["pitch"][:,1:]
	y_duration = data["duration"][:,1:]

	return x_pitch, y_pitch, x_duration, y_duration, x_mask, y_mask

