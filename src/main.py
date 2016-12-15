import numpy as np 
import models
from data import load_data, dataStatsBarPlot, write2table
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks', palette='Set2')
seaborn.set_context("paper")

def load_train_save_model(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, USE_DETERMINISTIC_PREVIUOS_OUTPUT, set_x_input_to_zero, BATCH_SIZE, NUM_EPOCHS, train_data, valid_data, test_data=None, song_data=None, data_pitch_map=None, data_duration_map=None, write2midi=False, plotLearningCurves=False, plotActivationSeq=False, in_dropout_p=0.2, out_dropout_p=0.5, use_l2_penalty=False):

	if USE_DETERMINISTIC_PREVIUOS_OUTPUT is not None:
		print("Setting up extended RNN model: {} with {} GRU".format(model_name, NUM_GRU_LAYER_UNITS))
		model = models.GRU_Network_Using_Previous_Output(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, set_x_input_to_zero, USE_DETERMINISTIC_PREVIUOS_OUTPUT, in_dropout_p, use_l2_penalty)
	else: 
		print("Setting up normal RNN model: {} with {} GRU".format(model_name, NUM_GRU_LAYER_UNITS))
		model = models.GRU_Network(model_name, max_seq_len, num_features_pitch, num_features_duration, NUM_GRU_LAYER_UNITS, set_x_input_to_zero, in_dropout_p, out_dropout_p, use_l2_penalty)

	model.load()
	if NUM_EPOCHS > 0:
		model.train(train_data, valid_data, NUM_EPOCHS, BATCH_SIZE)
		model.save()

	# Evaluate and printout results:
	if test_data is not None:
		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(test_data, None, None, False, data_pitch_map, data_duration_map, False, False)
		print("test_data accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))

	if song_data is not None:
		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(song_data, None, None, write2midi, data_pitch_map, data_duration_map, plotActivationSeq, False)
		print("song_data accuracy: (pitch, duration)=({:.3g},{:.3g})".format(float(acc_pitch), float(acc_duration)))
		print(np.argmax(output_pitch,axis=2))

	return model


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
	data_mask_y = data_mask[:,1:]

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
	all_data = {"pitch": data_pitch, "duration": data_duration, "mask": data_mask, "metadata": metadata, "indices": np.arange(data_pitch.shape[0])}

	# Evalute models on melody 202. Fiddle Hill Jig and not 706. Fred Roden's Reel
	song_idx = np.array(song_number)
	for i in range(song_idx.shape[0]):
		print("Evaluate models on melody {}. - {}:".format(song_idx[i], metadata[song_idx[i]][1][1]))
	song_data = {"pitch": data_pitch[song_idx,:,:], "duration": data_duration[song_idx,:,:], "mask": data_mask[song_idx], "metadata": metadata, "indices": song_idx}

	print(song_data["mask"])

	write2midi = False
	plotLearningCurves = True
	plotActivationSeq = False

	# Setting model feature shapes
	N_total, max_seq_len, num_features_pitch = np.shape(data_pitch)
	num_features_duration = np.shape(data_duration)[2]
	num_features_total = num_features_duration + num_features_pitch

	# figure handles for note-stat barplots
	fig_pitch, ax_pitch = plt.subplots(figsize=(9,6))
	fig_duration, ax_duration = plt.subplots(figsize=(6,6))

	n_colors = 10
 	palette = seaborn.color_palette(palette='muted', n_colors=n_colors, desat=None)
 	pitch_palette = seaborn.color_palette(palette='husl', n_colors=n_colors, desat=None) 
 	duration_palette = seaborn.color_palette(palette='husl', n_colors=n_colors, desat=0.5)

	# TODO: DATA STATS - WORK IN PROGRESS - PERCENTAGES
	rects_duration = dataStatsBarPlot(data_duration, data_mask, data_duration_map, palette, is_pitch=False, ax=ax_duration)
	fig_duration.canvas.draw()
	rects_pitch = dataStatsBarPlot(data_pitch, data_mask, data_pitch_map, palette, is_pitch=True, ax=ax_pitch)
	fig_pitch.canvas.draw()


	NUM_GRU_LAYER_UNITS, BATCH_SIZE, NUM_EPOCHS = 25, 10, 0


	# Initialize model specs.
	model_specs = []
	# model_specs.append({'name': 'Normal_GRU_Network', 'use_deterministic_output': None, 'zero_input': False, 'in_dropout_p': 0, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True})
	model_specs.append({'name': 'Normal_GRU_Network_0', 'use_deterministic_output': None, 'zero_input': False, 'in_dropout_p': 0, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	# model_specs.append({'name': 'Normal_GRU_Network_1_with_20p_dropout', 'use_deterministic_output': None, 'zero_input': False, 'in_dropout_p': 0.2, 'out_dropout_p': 0, 'num_epochs': 50, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	model_specs.append({'name': 'Normal_GRU_Network_with_50p_dropout', 'use_deterministic_output': None, 'zero_input': False, 'in_dropout_p': 0.5, 'out_dropout_p': 0.5, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	model_specs.append({'name': 'Normal_GRU_Network_with_l2', 'use_deterministic_output': None, 'zero_input': False, 'in_dropout_p': 0, 'out_dropout_p': 0, 'use_l2_penalty': True, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})

	model_specs.append({'name': 'GRU_using_deterministic_previous_output', 'use_deterministic_output': True, 'zero_input': False, 'in_dropout_p': 0, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	# model_specs.append({'name': 'GRU_using_deterministic_previous_output_with_20p_dropout', 'use_deterministic_output': True, 'zero_input': False, 'in_dropout_p': 0.2, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True})
	model_specs.append({'name': 'GRU_using_deterministic_previous_output_with_50p_dropout', 'use_deterministic_output': True, 'zero_input': False, 'in_dropout_p': 0.5, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotActivationSeq': False, 'plotLearningCurves': True})
	model_specs.append({'name': 'GRU_using_deterministic_previous_output_with_l2_and_50p_dropout', 'use_deterministic_output': True, 'zero_input': False, 'in_dropout_p': 0.5, 'out_dropout_p': 0, 'use_l2_penalty': True, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	model_specs.append({'name': 'GRU_using_only_deterministic_previous_output', 'use_deterministic_output': True, 'zero_input': True, 'in_dropout_p': 0, 'out_dropout_p': 0, 'num_epochs': 0, 'num_gru': 100, 'plotDataStats': True, 'testEvaluation': True, 'plotLearningCurves': True})
	# model_specs.append({'name': 'GRU_using_only_deterministic_previous_output_with_l2', 'use_deterministic_output': True, 'zero_input': True, 'in_dropout_p': 0, 'out_dropout_p': 0, 'use_l2_penalty': True, 'num_epochs': 200, 'num_gru': 100, 'plotDataStats': False, 'testEvaluation': False})




	# model_names.append({'name': 'GRU_using_nondeterministic_previous_output_with_50p_dropout', 'use_deterministic_output': False, 'zero_input': False, 'in_dropout_p': 0.5, 'out_dropout_p': 0, 'num_epochs': 200, 'num_gru': 100})
	# model_names.append({'name': 'GRU_using_nondeterministic_previous_output_with_20p_dropout', 'use_deterministic_output': False, 'zero_input': False, 'in_dropout_p': 0.2, 'out_dropout_p': 0, 'num_epochs': 200, 'num_gru': 100})

	plot_count = 0 

	# Setup and train 
	for i, model_spec in enumerate(model_specs):
		if 'plotLearningCurves' in model_spec:
			plotLearningCurves = model_spec['plotLearningCurves']
		else:
			plotLearningCurves = False

		if 'plotActivationSeq' in model_spec:
			plotActivationSeq = model_spec['plotActivationSeq']
		else:
			plotActivationSeq = False

		if 'testEvaluation' in model_spec:
			testEvaluation = model_spec['testEvaluation']
		else: 
			testEvaluation = False

		if 'write2midi' in model_spec:
			write2midi = model_spec['write2midi']
		else: 
			write2midi = False

		if 'plotDataStats' in model_spec:
			plotDataStats = model_spec['plotDataStats']
		else: 
			plotDataStats = False

		if 'use_l2_penalty' in model_spec:
			use_l2_penalty = model_spec['use_l2_penalty']
		else: 
			use_l2_penalty = False		

		model = load_train_save_model(model_spec['name'], max_seq_len, num_features_pitch, num_features_duration, model_spec['num_gru'], model_spec['use_deterministic_output'], model_spec['zero_input'], BATCH_SIZE, model_spec['num_epochs'], train_data, valid_data, test_data, song_data, data_pitch_map, data_duration_map, write2midi, plotLearningCurves, plotActivationSeq, model_spec['in_dropout_p'], model_spec['out_dropout_p'], use_l2_penalty)

		if plotLearningCurves:
			print "plotting learning curves of ", model_spec['name'], "\n"
			if plot_count > 0:
				fig_list = model.plotLearningCurves(pitch_palette[plot_count], duration_palette[plot_count], fig_list=fig_list, save_png_now=(plot_count>=6), model_num=plot_count+1)	
			else:
				fig_list = model.plotLearningCurves(pitch_palette[plot_count], duration_palette[plot_count], fig_list=None, save_png_now=(plot_count>=6), model_num=plot_count+1)
			plot_count += 1

		# Plot the histogram over model reconstructions for total data set.
		if plotDataStats:
			print "plotting data stats of ", model_spec['name'], "\n"
			cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(all_data, None, None, write2midi, data_pitch_map, data_duration_map, False, False)
			rects_duration = dataStatsBarPlot(output_duration, data_mask_y, data_duration_map, palette, is_pitch=False, ax=ax_duration, rects=rects_duration)
			fig_duration.canvas.draw()
			rects_pitch = dataStatsBarPlot(output_pitch, data_mask_y, data_pitch_map, palette, is_pitch=True, ax=ax_pitch, rects=rects_pitch)
			fig_pitch.canvas.draw()

		# Evaluate on test set
		cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration = model.evaluate(test_data, None, None, write2midi, data_pitch_map, data_duration_map, False, False)

		# Write evaluation measures to a LaTex table.
		if testEvaluation: 
			if model_spec['use_deterministic_output'] is not None: 
				if model_spec['use_deterministic_output']:
					model_num = 2
				else:
					model_num = 3
			else: 
				model_num = 1
			write2table('../data/eval_table', model_num, model_spec['in_dropout_p'], int(use_l2_penalty), cost_pitch, cost_duration, acc_pitch, acc_duration)


	# Plot the entire histogram
	legend_names = tuple(['original data'] + [mod['name'].replace('_',' ') + ' (#gru=' + str(mod['num_gru'])+')' for mod in model_specs])
	bars_pitch = tuple([rect[0] for rect in rects_pitch])
	bars_duration = tuple([rect[0] for rect in rects_duration])

	ax_pitch.legend(bars_pitch, legend_names)
	ax_duration.legend(bars_duration, legend_names)

	plt.figure(fig_pitch.number)
	plt.savefig("../data/models/" + "pitch_freq_barplot.png")
	plt.figure(fig_duration.number)
	plt.savefig("../data/models/" + "duration_freq_barplot.png")


if __name__ == '__main__':
	main()