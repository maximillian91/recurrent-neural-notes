import numpy as np 
import models
from data import load_data

def load_train_save_model(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, set_x_input_to_zero, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data=None, song_data=None, data_pitch_map=None, data_duration_map=None, write2midi=False, plotLearningCurves=False, in_dropout_p=0.2, out_dropout_p=0.5):

	if USE_DETERMINISTIC_PREVIUOS_OUTPUT is not None:
		print("Setting up extended RNN model: {} with {} GRU".format(model_name, NUM_GRU_LAYER_UNITS))
		model = models.GRU_Network_Using_Previous_Output(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, set_x_input_to_zero, USE_DETERMINISTIC_PREVIUOS_OUTPUT,in_dropout_p)
	else: 
		print("Setting up normal RNN model: {} with {} GRU".format(model_name, NUM_GRU_LAYER_UNITS))
		model = models.GRU_Network(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, in_dropout_p, out_dropout_p)
	# model.load(model0_name, 200)
	model.load()
	if NUM_EPOCHS > 0:
		model.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
		model.save()
	if plotLearningCurves:
		model.plotLearningCurves()

	# Evaluate and printout results:
	if test_data is not None:
		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map, False)
		print("test_data accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))

	if song_data is not None:
		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map, plotLearningCurves)
		print("song_data accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
		print(np.argmax(output_pitch,axis=2))



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

	song_number = [202, 450]#test_idx[2]
	for i, idx in enumerate([train_idx, valid_idx, test_idx]):
		if song_number in idx:
			print(i)

	# Partition data:
	train_data = {"pitch": data_pitch[train_idx], "duration": data_duration[train_idx], "mask": data_mask[train_idx], "metadata": metadata, "indices": train_idx}
	valid_data = {"pitch": data_pitch[valid_idx], "duration": data_duration[valid_idx], "mask": data_mask[valid_idx], "metadata": metadata, "indices": valid_idx}
	test_data = {"pitch": data_pitch[test_idx], "duration": data_duration[test_idx], "mask": data_mask[test_idx], "metadata": metadata, "indices": test_idx}

	# Evalute models on melody 202. Fiddle Hill Jig and not 706. Fred Roden's Reel
	song_idx = np.array(song_number)
	for i in range(song_idx.shape[0]):
		print("Evaluate models on melody {}. - {}:".format(song_idx[i], metadata[song_idx[i]][1][1]))
	song_data = {"pitch": data_pitch[song_idx,:,:], "duration": data_duration[song_idx,:,:], "mask": data_mask[song_idx], "metadata": metadata, "indices": song_idx}

	print(song_data["mask"])

	write2midi = False
	plotLearningCurves = False

	# Setting model feature shapes
	N_total, max_seq_len, num_features_pitch = np.shape(data_pitch)
	num_features_duration = np.shape(data_duration)[2]
	num_features_total = num_features_duration + num_features_pitch


	NUM_GRU_LAYER_UNITS, BATCH_SIZE, NUM_EPOCHS = 25, 10, 20

	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = True
	# set_x_input_to_zero = False

	# model4_name = "GRU_using_previous_output_only_4"
	# load_train_save_model(model4_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, True)

	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = False
	# set_x_input_to_zero = False

	# model5_name = "GRU_using_previous_output_only_5"
	# load_train_save_model(model5_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, True)

	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = False
	# set_x_input_to_zero = True

	# model6_name = "GRU_using_previous_output_only_6"
	# load_train_save_model(model6_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, True)

	# Setup and train GRU_using_only_deterministic_previous_output model
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = True
	# set_x_input_to_zero = False
	# model_name = "GRU_using_deterministic_previous_output"
	# for NUM_GRU_LAYER_UNITS in [25, 50, 75, 100]:
	# 	load_train_save_model(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, set_x_input_to_zero, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves)


	in_dropout_p = 0.2
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = True
	# set_x_input_to_zero = True
	# model_name = "GRU_using_deterministic_previous_output"
	# for NUM_GRU_LAYER_UNITS in [25, 50, 75, 100]:
	# 	load_train_save_model(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT,in_dropout_p, set_x_input_to_zero, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves)

	NUM_GRU_LAYER_UNITS = 50
	USE_DETERMINISTIC_PREVIUOS_OUTPUT = None
	set_x_input_to_zero = False
	in_dropout_p = 0.2
	out_dropout_p = 0.5
	model0_name = "Normal_GRU_Network_with_Dropout"
	load_train_save_model(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, set_x_input_to_zero, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves, in_dropout_p, out_dropout_p)

	# NUM_GRU_LAYER_UNITS = 50
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = None
	# set_x_input_to_zero = False
	# model0_name = "Normal_GRU_Network_0"
	# load_train_save_model(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves)

	# NUM_GRU_LAYER_UNITS = 75
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = None
	# set_x_input_to_zero = False
	# model0_name = "Normal_GRU_Network_0"
	# load_train_save_model(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves)


	# NUM_GRU_LAYER_UNITS = 100
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = None
	# set_x_input_to_zero = False
	# model0_name = "Normal_GRU_Network_0"
	# load_train_save_model(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, False, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves)



	# Setting model hyperparameters 
	# BATCH_SIZE = 10
	# NUM_GRU_LAYER_UNITS = 50
	# NUM_EPOCHS = 0
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = True

	# dropout_range = np.arange(5, max_seq_len-1, 1)
	# dropout_fraction = 0.25

	# ###### MODELS ########
	# model0_name = "Normal_GRU_Network"
	# print("MODEL: {} with {} GRU".format(model0_name, NUM_GRU_LAYER_UNITS))
	# model0 = models.GRU_Network(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS)
	# model0.load(model0_name, 200)
	# # model0.load(model0_name)	
	# # model0.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	# # model0.plotLearningCurves()
	# # model0.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# # cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# # print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# # cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# # print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))


	# model1_name = "GRU_using_previous_output"
	# print("MODEL: {} with {} GRU".format(model1_name, NUM_GRU_LAYER_UNITS))
	# model1 = models.GRU_Network_Using_Previous_Output(model1_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT)
	# model1.load(model1_name, 200)
	# # model1.load(model1_name)
	# # model1.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	# # model1.plotLearningCurves()
	# # model1.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))


	# model2_name = "GRU_using_nondeterministic_previous_output"
	# print("MODEL: {} with {} GRU".format(model2_name, NUM_GRU_LAYER_UNITS))
	# model2 = models.GRU_Network_Using_Previous_Output(model2_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, False)
	# model2.load(model2_name, 200)
	# # model2.load(model2_name)	
	# # model2.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)	
	# # model2.plotLearningCurves()
	# # model2.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))


	# Setting model hyperparameters 
	# BATCH_SIZE = 10
	# NUM_GRU_LAYER_UNITS = 25
	# NUM_EPOCHS = 0
	# USE_DETERMINISTIC_PREVIUOS_OUTPUT = True


	# ###### MODELS ########
	# model0_name = "Normal_GRU_Network"
	# print("MODEL: {} with {} GRU".format(model0_name, NUM_GRU_LAYER_UNITS))
	# model0 = models.GRU_Network(model0_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS)
	# model0.load(model0_name, 200)
	# # model0.load(model0_name)	
	# # model0.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	# # model0.plotLearningCurves()
	# # model0.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model0.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	

	# model1_name = "GRU_using_previous_output"
	# print("MODEL: {} with {} GRU".format(model1_name, NUM_GRU_LAYER_UNITS))
	# model1 = models.GRU_Network_Using_Previous_Output(model1_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT)
	# model1.load(model1_name, 200)
	# # model1.load(model1_name)
	# # model1.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
	# # model1.plotLearningCurves()
	# # model1.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model1.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))


	# model2_name = "GRU_using_nondeterministic_previous_output"
	# print("MODEL: {} with {} GRU".format(model2_name, NUM_GRU_LAYER_UNITS))
	# model2 = models.GRU_Network_Using_Previous_Output(model2_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, False)
	# model2.load(model2_name, 200)
	# # model2.load(model2_name)	
	# # model2.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)	
	# # model2.plotLearningCurves()
	# # model2.save()
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map)
	# print("test_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, dropout_range, None, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, dropout_range, None accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
	# cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration =model2.evaluate(song_data, None, dropout_fraction, write2midi, data_pitch_map, data_duration_map)
	# print("song_data, None, dropout_fraction accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))



if __name__ == '__main__':
	main()