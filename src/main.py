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
	data_pitch_map = data["pitch"]["map_ind2feat"]
	data_duration = data["duration"]["encoded"]
	data_duration_map = data["duration"]["map_ind2feat"]
	data_mask = data["mask"]
	metadata = data_orig["metadata"]

	train_idx = data["train_idx"]
	valid_idx = data["valid_idx"]
	test_idx = data["test_idx"]

	# Partition data:
	train_data = {"pitch": data_pitch[train_idx], "duration": data_duration[train_idx], "mask": data_mask[train_idx]}
	valid_data = {"pitch": data_pitch[valid_idx], "duration": data_duration[valid_idx], "mask": data_mask[valid_idx]}
	test_data = {"pitch": data_pitch[test_idx], "duration": data_duration[test_idx], "mask": data_mask[test_idx]}

	# Evalute models on melody 202. Fiddle Hill Jig:
	song_idx = np.array([202,700])
	for i in song_idx:
		print("Evalute models on melody {}. - {}:".format(i, metadata[i][1][1]))
	song_data = {"pitch": data_pitch[song_idx], "duration": data_duration[song_idx], "mask": data_mask[song_idx,:], "metadata": metadata, "indices": song_idx}
	write2midi = True

	# Setting model feature shapes
	N_total, max_seq_len, num_features_pitch = np.shape(data_pitch)
	num_features_duration = np.shape(data_duration)[2]
	num_features_total = num_features_duration + num_features_pitch

	# Setting model hyperparameters 
	BATCH_SIZE = 10
	NUM_GRU_LAYER_UNITS = 50
	NUM_EPOCHS = 190
	USE_DETERMINISTIC_PREVIUOS_OUTPUT = True


	###### MODELS ########
	model0_name = "Normal_GRU_Network"
	model0 = models.GRU_Network(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS)
	model0.load(model0_name)	
	#model0.load(model0_name, 200)
	model0.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	model0.plotLearningCurves()
	model0.save()
	#model0.evaluate(song_data, write2midi, data_pitch_map, data_duration_map)

	model1_name = "GRU_using_previous_output"
	model1 = models.GRU_Network_Using_Previous_Output(model1_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT)
	model1.load(model1_name)
	#model1.load(model1_name, 200)
	model1.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	model1.plotLearningCurves()
	model1.save()
	model1.evaluate(song_data, write2midi, data_pitch_map, data_duration_map)


	model2_name = "GRU_using_nondeterministic_previous_output"
	model2 = models.GRU_Network_Using_Previous_Output(model2_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, False)
	model2.load(model2_name)	
	#model2.load(model2_name, 200)
	model2.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)	
	model2.plotLearningCurves()
	model2.save()
	#model2.evaluate(song_data, write2midi, data_pitch_map, data_duration_map)




if __name__ == '__main__':
	main()