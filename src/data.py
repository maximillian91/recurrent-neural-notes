import numpy as np
import random
import string
from pprint import pprint
import cPickle as pickle
import os.path
from os import listdir
from music21 import converter, note, stream, duration, midi
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks', palette='Set2')
seaborn.set_context("paper")


def plotHeatMap(data_set, name):
	figure = plt.figure()
	axis = figure.add_subplot(1, 1, 1)
	seaborn.heatmap(data_set.T, xticklabels = False, yticklabels = False, cbar = True, square = True, ax = axis)
	figure_name = name + "_heatmap.png"
	plt.savefig(figure_name)

def save_data_from_abc_files(data_file="data"):	
	# Data collection: For saving and returning the abc-data files in a dict of list of lists. 

	# Convert abc to music21 object
	abc_files = [file for file in listdir("../data/") if (file[0] != "." and file.split(".")[-1] == "abc")]
	non_reel_files = [file for file in abc_files if file[0:5] != "reels"]
	reel_files = [file for file in abc_files if file[0:5] == "reels"]

	#abc_files.pop(6) # pop out broken file without default quarterlength
	# Only 2 scores (deleted in .abc) have been excluded due to wrong formatting, so parse couldn't read them.
	print("Found these .abc files:\n{}".format(abc_files))
	data_path = "../data/"
	pkl_path = "../data/pkl/"
	file_to_combine = "reels"
	data_file_path = pkl_path + data_file + ".pkl"

	data = {}
	data["pitch"] = []
	data["duration"] = []
	data["metadata"] = []
	data["file_slices"] = {}

	# Importing all files except reels.	
	start_index = 0
	for abc_file in non_reel_files:
	    print("\nCollecting from {}".format(abc_file))
	    abc_file_path = data_path + abc_file
	    opus = converter.parse(abc_file_path)
	    pitch_opus = [] # midi values for notes in all scores in opus
	    duration_opus = [] # float durations for notes in all scores in opus
	    metadata_opus = [] # metadata for notes in all scores in opus
	    print("with length={}".format(len(opus)))

	    for n, score in enumerate(opus):
	        print("Score #{}: {}".format(n,score.metadata.title))
	        # Lists for all notes and rests in the score s
	        pitch_score = []
	        duration_score = []
	        for n in score.flat.notesAndRests:
	            if type(n) is note.Note:
	                pitch_score.append(n.pitch.midi)
	                duration_score.append(float(n.duration.quarterLength))
	            elif type(n) is note.Rest:
	                pitch_score.append(-1.0) # we set the midi-pitch-code for rests to 1000 (greatest pitch value = last bit in 1-hot-encoded vector)
	                duration_score.append(float(n.duration.quarterLength))
	        # We set last step in pitch and duration to -1 for end-of-sequence marker (<eos>). e.g. min pitch value = first bit in 1-hot-encoded vector
	        pitch_score.append(-2.0)
	        duration_score.append(0.0)

	        # Add the score (example) to the list of scores.
	        pitch_opus.append(pitch_score)
	        duration_opus.append(duration_score)
	        metadata_opus.append(score.metadata.all())

	    # Concatenate list of scores to the rest of the list of scores in data dictionary. 
	    end_index = start_index + len(pitch_opus)
	    file_slice = slice(start_index,end_index)
	    start_index = end_index  
	    data["pitch"] += pitch_opus
	    data["duration"] += duration_opus
	    data["metadata"] += metadata_opus
	    data["file_slices"][abc_file.split(".")[0]] = file_slice 


	# Importing reel files
	pitch_opus = [] # midi values for notes in all scores in opus
	duration_opus = [] # float durations for notes in all scores in opus
	metadata_opus = [] # metadata for notes in all scores in opus
	for abc_file in reel_files:
	    print("Collecting from {}:".format(abc_file))
	    abc_file_path = data_path + abc_file
	    opus = converter.parse(abc_file_path)
	    for score in opus:
	        print(score.metadata.title)
	        # Lists for all notes and rests in the score s
	        pitch_score = []
	        duration_score = []
	        for n in score.flat.notesAndRests:
	            if type(n) is note.Note:
	                pitch_score.append(n.pitch.midi)
	                duration_score.append(float(n.duration.quarterLength))
	            elif type(n) is note.Rest:
	                pitch_score.append(1000) # we set the midi-pitch-code for rests to 1000 (greatest pitch value = last bit in 1-hot-encoded vector)
	                duration_score.append(float(n.duration.quarterLength))
	        # We set last step in pitch and duration to -1 for end-of-sequence marker (<eos>). e.g. min pitch value = first bit in 1-hot-encoded vector
	        pitch_score.append(-2.0)
	        duration_score.append(0.0)

	        pitch_opus.append(pitch_score)
	        duration_opus.append(duration_score)
	        metadata_opus.append(score.metadata.all())

	end_index = start_index + len(pitch_opus)
	file_slice = slice(start_index,end_index)
	data["pitch"] += pitch_opus
	data["duration"] += duration_opus
	data["metadata"] += metadata_opus
	data["file_slices"]["reels"] = file_slice


	with open(data_file_path, "wb") as file:
	    pickle.dump(data, file)

	return data
#
def sample_data(n_examples,max_n_timesteps,max_n_of_features_pr_timestep):
	examples = []
	for n in range(n_examples):
		example = []
		n_timesteps = random.randrange(0,max_n_timesteps)
		for t in range(n_timesteps):
			timestep = []
			n_features = random.randrange(1,max_n_of_features_pr_timestep)
			for c in range(n_features):
				timestep.append(draw_random_feature())
			example.append(timestep)
		examples.append(example)
	return examples


def draw_random_feature():
	r = random.randrange(26)
	n = random.randrange(100)
	d = random.randrange(10)
	feature = "{}{:02d}.{}".format(chr(r+65),n,d)  
	return feature

def unique_feature_set_mapping(examples):
	features = set()
	for example in examples:
		features.update(example)

	features = sorted(features)
	feat2ind = dict()
	ind2feat = dict()
	for i, feature in enumerate(features):
		feat2ind[feature] = i
		ind2feat[i] = feature

	return feat2ind, ind2feat

def one_hot_encoder(examples):
	N = len(examples)
	MAX_SEQ_LEN = len(max(examples,key=len))
	feat2ind, ind2feat = unique_feature_set_mapping(examples)
	NUM_FEATURES = len(feat2ind)
	eos_value = examples[0][-1]
	print("<eos> value: {}".format(eos_value))


	# One-hot-encoding step which returns 3D numpy arrays of dimensions:
		# dim(input_pitch) = (N, MAX_SEQ_LEN, NUM_FEATURES)
	examples_ohe = np.zeros((N, MAX_SEQ_LEN, NUM_FEATURES)).astype("int32")
	examples_mask = np.zeros((N, MAX_SEQ_LEN)).astype("float32")
	examples_ind = np.zeros((N, MAX_SEQ_LEN)).astype("int32")
	examples_pad = eos_value*np.ones((N, MAX_SEQ_LEN)).astype("float32")

	for i in range(N):
		for j in range(MAX_SEQ_LEN):
			if j >= len(examples[i]):
				t = eos_value
			else:
				t = examples[i][j]
				examples_mask[i,j] = 1.0
			examples_ohe[i,j,feat2ind[t]] = 1
			examples_ind[i,j] = feat2ind[t]
			examples_pad[i,j] = t
		# examples_mask[i,len(x)-1] = 0
	return examples_ohe, examples_pad, examples_ind, examples_mask, feat2ind, ind2feat

def one_hot_decoder(examples_encoded, map_ind2feat):
	# Examples are not one-hot-encoded, but passed after argmax
	# map_ind2feat is the dictionary mapping from argmax-index to the original feature value.
	examples_shape = examples_encoded.shape
	print(len(examples_shape))
	if len(examples_shape) < 3:
		N_total = 1
		MAX_SEQ_LEN = examples_shape[0]
	else:
		N_total = examples_shape[0]
		MAX_SEQ_LEN = examples_shape[1]

	examples_encoded = examples_encoded.reshape((N_total, MAX_SEQ_LEN, -1))
	examples_ind = np.argmax(examples_encoded, axis=-1)
	examples = np.zeros((N_total, MAX_SEQ_LEN))

	for i in range(N_total):
		for j in range(MAX_SEQ_LEN):
			#examples[i,j] = map_ind2feat[examples_feat_ind[i,j]]
			examples[i,j] = map_ind2feat[examples_ind[i,j]]
			#example.append(np.nonzero(examples_ohe[i,j,:])[0].tolist())
	return examples

def array2streams(pitch, pitch_map, duration, duration_map):
	# Convert numpy arrays to lists with original features
	pitch_decoded = one_hot_decoder(pitch, pitch_map)
	duration_decoded = one_hot_decoder(duration, duration_map)

	# Convert lists to music21 stream
	N, M = pitch_decoded.shape
	melodies = []
	for i in range(N):
		melody = stream.Stream()
		for j in range(M):
			if pitch_decoded[i,j] < -1:
				continue
			elif pitch_decoded[i,j] == -1:
				n = note.Rest()
			else: 
				n = note.Note()
				n.pitch.midi = pitch_decoded[i,j]
			n.duration.quarterLength = duration_decoded[i,j]
			melody.append(n)
		melodies.append(melody)
	return melodies

def array2midi(pitch, pitch_map, duration, duration_map, metadata, indices, filepath="../data/models/", filename="original"):
	# Convert and save stream to midi file
	melodies = array2streams(pitch, pitch_map, duration, duration_map)
	for i, melody in enumerate(melodies):
		mf = midi.translate.streamToMidiFile(melody)
		mf.open(filepath + "melody_{}_{}".format(metadata[indices[i]][1][1], filename) + '.mid','wb')
		mf.write()
		mf.close()


def plotActivations(name, activations, mask, pitch, pitch_map, duration, duration_map, metadata, indices):
	melodies = array2streams(pitch, pitch_map, duration, duration_map)
	#seaborn.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
	seaborn.set(font_scale=0.5)
	for i, melody in enumerate(melodies):
		# figure = plt.figure(figsize=(8,6))
		# axis = figure.add_subplot(1, 1, 1)
		activations_trimmed = activations[i,(mask[i,:] > 0),:]
		# column_labels = [x.pitch.name if x.isNote else "%" for x in melody.elements]
		# #column_labels = [x.pitch.nameWithOctave + "\n" + str(x.duration.quarterLength) if x.isNote else "0" + "\n" + str(x.duration.quarterLength) for x in melody.elements]
		# row_labels = [str(x) for x in range(activations_trimmed.shape[-1])]
		# seaborn.heatmap(activations_trimmed.T, xticklabels = column_labels, yticklabels = True, cbar = True, square = True, ax = axis)

		#labels
		#column_labels = [x.pitch.nameWithOctave + "\n" + str(x.duration.quarterLength) if x.isNote else "0" + "\n" + str(x.duration.quarterLength) for x in melody.elements]

		#axis.set_xticklabels(column_labels, minor=True)
		#axis.set_yticklabels(row_labels, minor=True)
		figure_name = name + "_" + metadata[indices[i]][1][1] + "_gruActivations"
		# plt.savefig(figure_name + "_heatmap.png")
		# plt.clf()

		Stream2Img(figure_name, melody, activations_trimmed)




def Stream2Img(name, stream, activations):
	#activations = np.random.rand(len(streams), len(streams[0]))

	MAX_SEQ_LEN, NUM_GRU = activations.shape

	palette = seaborn.diverging_palette(240, 10, l=75, n=100)
	activations_binned_idx = (activations*100).astype("int")

	NOTES_PER_LINE = 16
	row_pixel_start = 10
	col_pixel_start = 10
	newline_space = 5
	# get a font
	fnt = ImageFont.truetype('System/Library/Fonts/Menlo.ttc', 12)

	for gru in range(NUM_GRU):
		txt = Image.new('RGBA', (600,500), (255,255,255,0))
		# get a drawing context
		d = ImageDraw.Draw(txt)
		string_width, string_height = d.textsize("CCCCC", font=fnt)
		row_pixel = row_pixel_start
		for j, note in enumerate(stream):
			if note.isNote:
				pitch_string = "{:^5}".format(note.pitch.nameWithOctave)
			else:
				pitch_string = "  %  "
			if note.duration.quarterLength < 4:
			    duration_string = " 1/{:<2.0f}".format(4/note.duration.quarterLength)
			elif note.duration.quarterLength == 8.0:
			    duration_string = " 2/1 "
			elif note.duration.quarterLength == 6.0:
			    duration_string = " 3/2 "
			else:
			    duration_string = note.duration.quarterLength
			col_pixel = col_pixel_start + (j%NOTES_PER_LINE)*string_width

			color = tuple([int(255*p) for p in palette[activations_binned_idx[j,gru]].tolist()])
			d.rectangle([col_pixel, row_pixel, col_pixel+string_width, row_pixel+(2*string_height + 5)], fill=color)
			d.text((col_pixel, row_pixel), pitch_string, font=fnt, fill=(1,1,1,255))
			# draw text, full opacity
			d.text((col_pixel, row_pixel+15), duration_string, font=fnt, fill=(1,1,1,255))

			if (j+1) % NOTES_PER_LINE == 0:
				# draw text, half opacity
				row_pixel += (2*string_height + 15 + newline_space)
		txt.save(name + "_gru#" + str(gru+1) + ".png")




def save_partitioning(data=None, data_file="data", partition_file="partition", train_partition=0.8):
 	# path for pickled files:
	pkl_path = "../data/pkl/"
	data_file_path = pkl_path + data_file + ".pkl" 
	partition_file_path = pkl_path + partition_file + ".pkl"

	if data is None:
		if os.path.isfile(data_file_path):
			print("Collecting data from {}".format(data_file_path))
			with open(data_file_path, "rb") as file:
				data = pickle.load(file)

	N_total = len(data["pitch"])

	indices = np.random.permutation(N_total)

	N_part = int(N_total*train_partition)
	N_test = N_total - N_part
	N_train = int(N_part*train_partition)
	N_valid = N_part - N_train

	train_indices = indices[:N_train]
	valid_indices = indices[N_train:N_part]
	test_indices = indices[N_part:]

	if (len(train_indices) + len(valid_indices) + len(test_indices)) == N_total:
		data_ind = {"train_idx": train_indices, "valid_idx": valid_indices, "test_idx": test_indices}
		print("Dataset was split properly into partitions:\n\t#train={}\n\t#valid={}\n\t#test={}\n\t#total={}".format(N_train, N_valid, N_test, N_total))
		with open(partition_file_path, "wb") as file:
			pickle.dump(data_ind, file)  
	else: 
		print("Warning! Partitioning was not saved due to improper splitting:\n\t#train={}\n\t#valid={}\n\t#test={}\n\t#total={}".format(N_train, N_valid, N_test, N_total))
	return data_ind


def load_data(data_file="data", partition_file="partition", train_partition=0.8):
 	# path for pickled data files:
	pkl_path = "../data/pkl/"
	data_file_path = pkl_path + data_file + ".pkl" 
	partition_file_path = pkl_path + partition_file + ".pkl"

	if os.path.isfile(data_file_path):
		print("Collecting data from {}".format(data_file_path))
		with open(data_file_path, "rb") as file:
			data = pickle.load(file)
	else: 
		data = save_data_from_abc_files(data_file)


	if os.path.isfile(partition_file_path):
		print("Collecting data partitioning from {}".format(partition_file_path))
		with open(partition_file_path, "rb") as file:
			partition = pickle.load(file)
	else: 
		print("{} does not exist, so let me just create it.".format(partition_file_path))
		partition = save_partitioning(data, data_file, partition_file, 0.8)

	MAX_SEQ_LEN = len(max(data["pitch"],key=len))
	print("Maximum length of notes in all melodies are {}".format(MAX_SEQ_LEN))
	# One-hot-encoding step which returns 3D numpy arrays of dimensions:
		# dim(input_pitch) = (N_total, MAX_SEQ_LEN, NUM_PITCHES)
		# dim(input_duration) = (N_total, MAX_SEQ_LEN, NUM_DURATIONS) 
	x_pitch_ohe, x_pitch, x_pitch_ind, x_pitch_mask, feat2ind_pitch, ind2feat_pitch = one_hot_encoder(data["pitch"])
	x_duration_ohe, x_duration, x_duration_ind, x_duration_mask, feat2ind_duration, ind2feat_duration = one_hot_encoder(data["duration"])

	data_ohe = {}
	data_ohe["train_idx"] = partition["train_idx"]
	data_ohe["valid_idx"] = partition["valid_idx"]
	data_ohe["test_idx"] = partition["test_idx"]
	data_ohe["pitch"] = {"encoded": x_pitch_ohe, "original": x_pitch, "indices": x_pitch_ind, "map_ind2feat": ind2feat_pitch, "map_feat2ind": feat2ind_pitch}
	data_ohe["duration"] = {"encoded": x_duration_ohe, "original": x_duration, "indices": x_duration_ind, "map_ind2feat": ind2feat_duration, "map_feat2ind": feat2ind_duration}
	data_ohe["mask"] = x_pitch_mask
	data_ohe["metadata"] = data["metadata"]
	data_ohe["file_slices"] = data["file_slices"]

	return data_ohe, data

def analyze_results(result_file_path):
	if os.path.isfile(result_file_path):
		print("Collecting data from {}".format(result_file_path))
		with open(result_file_path, "rb") as file:
			results = pickle.load(file)
	elif os.path.isfile("../models/" + result_file_path):
		with open("../models/" + result_file_path, "rb") as file:
			results = pickle.load(file)
	else: 
		print("No such result file exists!")
		return "No such result file exists!"



	max_prob_pitch_recon = np.argmax(test_output_pitch,axis=2)
	max_prob_duration_recon = np.argmax(test_output_duration,axis=2)
	max_prob_pitch_orig = np.argmax(y_pitch_test,axis=2)
	max_prob_duration_orig = np.argmax(y_duration_test,axis=2)

	# Convert 3 first examples to the original feature values:
	pitch_decoded_recon = one_hot_decoder(max_prob_pitch_recon[0:3], data["pitch"]["map_ind2feat"])
	duration_decoded_recon = one_hot_decoder(max_prob_duration_recon[0:3], data["duration"]["map_ind2feat"]) 

	pitch_decoded_orig = one_hot_decoder(max_prob_pitch_orig[0:3], data["pitch"]["map_ind2feat"])
	duration_decoded_orig = one_hot_decoder(max_prob_duration_orig[0:3], data["duration"]["map_ind2feat"]) 

	# Write 3 first examples to midi files:
	array2midi(pitch_decoded_recon, duration_decoded_recon, data["metadata"][0:3], filepath=data_path, filename="recon")
	array2midi(pitch_decoded_orig, duration_decoded_orig, data["metadata"][0:3], filepath=data_path, filename="orig")



	#y_pitch_test_original = one_hot_decoder(test_output_pitch, map_ind2feat=data["map_ind2feat"])

	print("Inspect the first {} melodies:".format(number_of_test_examples))
	for i in range(number_of_test_examples):
		print("Pitch targets and prediction")
		print(max_prob_pitch_orig[i])
		print(max_prob_pitch_recon[i])

		print("Duration targets and prediction")
		print(max_prob_duration_orig[i])
		print(max_prob_duration_recon[i])


	# Accuracy plots
	# Pitch
	plt.figure()
	acc_train_pitch_plt, = plt.plot(acc_train_pitch)
	acc_valid_pitch_plt, = plt.plot(acc_valid_pitch)

	plt.legend([acc_train_pitch_plt, acc_valid_pitch_plt], ['Training', 'Validation'])

	plt.ylabel('Pitch Accuracy', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "acc_pitch.png")


	## Duration 
	plt.figure()
	acc_train_duration_plt, = plt.plot(acc_train_duration)
	acc_valid_duration_plt, = plt.plot(acc_valid_duration)

	plt.legend([acc_train_duration_plt, acc_valid_duration_plt], ['Training', 'Validation'])

	plt.ylabel('Duration Accuracy', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "acc_duration.png")


	# Cost plots
	## Accuracy
	plt.figure()
	cost_train_pitch_plt, = plt.plot(cost_train_pitch)
	cost_valid_pitch_plt, = plt.plot(cost_valid_pitch)

	plt.legend([cost_train_pitch_plt, cost_valid_pitch_plt], ['Training', 'Validation'])

	plt.ylabel('Pitch Crossentropy Cost', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "cost_pitch.png")


	## Duration
	plt.figure()
	cost_train_duration_plt, = plt.plot(cost_train_duration)
	cost_valid_duration_plt, = plt.plot(cost_valid_duration)

	plt.legend([cost_train_duration_plt, cost_valid_duration_plt], ['Training', 'Validation'])

	plt.ylabel('Duration Crossentropy Cost', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "cost_duration.png")



	# Plotting norms of weights
	## Horizontal weights
	plt.figure()
	horz_update_plt, = plt.plot(horz_update)
	horz_reset_plt, = plt.plot(horz_reset)
	horz_hidden_plt, = plt.plot(horz_hidden)

	plt.legend([horz_update_plt, horz_reset_plt, horz_hidden_plt], ['Updategate', 'Resetgate', 'Hidden Updategate'])
	plt.ylabel('Frobenius norm', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('Horizontal GRU weights', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "horz_weight_norms.png")

	## Vertical weights
	plt.figure()
	vert_update_plt, = plt.plot(vert_update)
	vert_reset_plt, = plt.plot(vert_reset)
	vert_hidden_plt, = plt.plot(vert_hidden)

	plt.legend([vert_update_plt, vert_reset_plt, vert_hidden_plt], ['Updategate', 'Resetgate', 'Hidden Updategate'])
	plt.ylabel('Frobenius norm', fontsize=15)
	plt.xlabel('Epoch #', fontsize=15)
	plt.title('Vertical GRU weights', fontsize=20)
	plt.grid('on')
	plt.savefig(fig_path + "vert_weight_norms.png")

def main():
	# path for pickled data files:
	# pkl_path = "../data/pkl/"
	# data_file_path = pkl_path + data_file + ".pkl"
	# model_data_path = "../data/models/"
	# model_name = "GRU_using_previous_output"

	# num_gru_layer_units, batch_size, number_of_epochs_trained = 50, 10, 10
	# model_loaded = False
	# ### LOAD model ###
	# model_name_spec = model_name + "_gru_{}_bs_{}_e_{}".format(num_gru_layer_units, batch_size, number_of_epochs_trained)
	# model_epochs = [int(file.split(".")[0].split("_")[-1]) for file in listdir(model_data_path) if (file[0] != "." and file[:len(model_name_spec)] == model_name_spec and file.split(".")[-1] == "pkl")]
	
	# # Check for latest model data
	# if model_epochs:
	# 	max_epoch_num = max(model_epochs)
	# 	print("The current number of epochs the {} model have been trained is: {}".format(model_name, max_epoch_num))
	# 	print("Loading the data for the current state of the model.")
	# 	model_path = model_data_path + model_name_spec + ".pkg"
	# 	if os.path.isfile(model_path):
	# 		model_name = model_name
	# 		print("Setting up model with previous parameters from the file {}".format(model_path))
	# 		with open(model_path, "rb") as file:
	# 			model = pickle.load(file)
	# 		model_loaded = True
	# else: 
	# 	print("No previous data on this model exists. Use the methods train() and save() first and then load().")


	data_ohe, data = load_data(data_file="data_new", partition_file="partition", train_partition=0.8)

	# for i, metadata in enumerate(data["metadata"]):
	# 	print("{}. - {}".format(i, metadata[1][1]))

	# pitch_decoded = one_hot_decoder(data_ohe["pitch"]["indices"], data_ohe["pitch"]["map_ind2feat"])
	# duration_decoded = one_hot_decoder(data_ohe["duration"]["indices"], data_ohe["duration"]["map_ind2feat"])

	streams = array2streams(data_ohe["pitch"]["encoded"][0,:,:], data_ohe["pitch"]["map_ind2feat"], data_ohe["duration"]["encoded"][0,:,:], data_ohe["duration"]["map_ind2feat"])

	Stream2Img(streams)

	# pitch_decoded = one_hot_decoder(data_ohe["pitch"]["indices"], data_ohe["pitch"]["map_ind2feat"])
	# duration_decoded = one_hot_decoder(data_ohe["duration"]["indices"], data_ohe["duration"]["map_ind2feat"])

	# array2midi(pitch_decoded[0:2], duration_decoded[0:2], data["metadata"], filename="original")
	# # print("Number of tunes in collection: {}".format(len(x_pitch)))
	# # print(len(x_pitch[data_slices["jigs"]]))
	# # #print(x_duration[data_slices["reels"]])
	
	# # Printout Test of our one-hot-encoded (OHE) data structure 
	# for s in range(2):
	# 	print("\n{}. {}".format(s+1, data["metadata"][s]))
	# 	for i in range(2):
	# 		print("Note {}:".format(i+1))
	# 		print("\tPitch:\n\t\tdecoded={}\toriginal={}".format(pitch_decoded[s,i], data["pitch"][s][i]))
	# 		print("\tDuration:\n\t\tdecoded={}\toriginal={}".format(duration_decoded[s,i], data["duration"][s][i]))


if __name__ == '__main__':
	main()