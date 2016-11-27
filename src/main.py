import numpy as np 
import models
from data import load_data

def main():
	data_file="data_new"
	partition_file="partition"
	train_partition=0.8

	# Importing data
	data, data_orig = load_data(data_file=data_file, partition_file=partition_file, train_partition=train_partition)

	data_pitch = data["pitch"]["encoded"]
	data_duration = data["duration"]["encoded"]
	data_mask = data["mask"]

	train_idx = data["train_idx"]
	valid_idx = data["valid_idx"]
	test_idx = data["test_idx"]

	# Partition data:
	train_data = {"pitch": data_pitch[train_idx], "duration": data_duration[train_idx], "mask": data_mask[train_idx]}
	valid_data = {"pitch": data_pitch[valid_idx], "duration": data_duration[valid_idx], "mask": data_mask[valid_idx]}
	test_data = {"pitch": data_pitch[test_idx], "duration": data_duration[test_idx], "mask": data_mask[test_idx]}

	# Setting model feature shapes
	N_total, max_seq_len, num_features_pitch = np.shape(data_pitch)
	num_features_duration = np.shape(data_duration)[2]
	num_features_total = num_features_duration + num_features_pitch

	# Setting model hyperparameters 
	BATCH_SIZE = 10
	NUM_GRU_LAYER_UNITS = 50
	NUM_EPOCHS = 10
	USE_DETERMINISTIC_PREVIUOS_OUTPUT = True


	###### MODELS ########
	# model0_name = "Normal_GRU_Network"
	# model0 = models.GRU_Network(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS)

	# model0.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)

	# model0.saveModel()


	model1_name = "GRU_using_previous_output"
	model1 = models.GRU_Network_Using_Previous_Output(model1_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT)
	model1.load(model1_name)
	model1.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)

	model1.save()

	# model2_name = "GRU_using_nondeterministic_previous_output"
	# model2 = models.GRU_Network_Using_Previous_Output(model2_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT)
	# #model2.loadModel()
	# model2.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)

	# model2.saveModel()




	# Defining model path
	
	#fig_path = "../models/fig/{}_gru_{}_bs_{}_e_{}_".format(model1_name, num_gru_layer_units, BATCH_SIZE, N_epochs) 
	#data_path = "../data/models/{}_gru_{}_bs_{}_e_{}_".format(model1_name, num_gru_layer_units, BATCH_SIZE, N_epochs) 

	#model2 = models.GRU_Network(max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS) 

	
	# Reconstructions
		# if self.number_of_epochs_trained()
		# model["train_recon_pitch"] = np.argmax(train_output_pitch, axis=2)
		# model["valid_recon_pitch"] = np.argmax(valid_output_pitch, axis=2)
		# model["test_recon_pitch"] = np.argmax(test_output_pitch, axis=2)

		# model["train_recon_duration"] = np.argmax(train_output_duration, axis=2)
		# model["valid_recon_duration"] = np.argmax(valid_output_duration, axis=2)
		# model["test_recon_duration"] = np.argmax(test_output_duration, axis=2)

if __name__ == '__main__':
	main()