import numpy as np
import random
import string
from pprint import pprint
import cPickle as pickle
import os.path
from os import listdir
from music21 import converter, note, stream, duration, midi

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
			examples_ohe[i,j,feat2ind[t]] = 1
			examples_mask[i,j] = 1
			examples_ind[i,j] = feat2ind[t]
			examples_pad[i,j] = t
		# examples_mask[i,len(x)-1] = 0
	return examples_ohe, examples_pad, examples_ind, examples_mask, feat2ind, ind2feat

def one_hot_decoder(examples_ind, map_ind2feat):
	# Examples are not one-hot-encoded, but passed after argmax
	# map_ind2feat is the dictionary mapping from argmax-index to the original feature value.
	examples_shape = np.shape(examples_ind)
	N_total = examples_shape[0]
	MAX_SEQ_LEN = examples_shape[1]
	#NUM_FEATURES = examples_shape[2]

	# Outcommented 
	#examples_flattened = examples_ohe.flatten() 
	#examples_nonzero = np.nonzero(examples_flattened)[0]
	#examples_feat_ind = np.mod(examples_nonzero,NUM_FEATURES).reshape(N_total, MAX_SEQ_LEN)

	examples = np.zeros((N_total, MAX_SEQ_LEN))

	for i in range(N_total):
		for j in range(MAX_SEQ_LEN):
			#examples[i,j] = map_ind2feat[examples_feat_ind[i,j]]
			examples[i,j] = map_ind2feat[examples_ind[i,j]]
			#example.append(np.nonzero(examples_ohe[i,j,:])[0].tolist())
	return examples

def array2midi(pitch, duration, metadata, filename="original"):
	# Convert numpy arrays to lists
	#pitch_list = pitch.tolist()
	#duration_list = duration.tolist()

	# Convert lists to music21 stream
	N, M = pitch.shape

	for i in range(N):
		melody = stream.Stream()
		for j in range(M):
			if pitch[i,j] < -1:
				continue
			elif pitch[i,j] == -1:
				n = note.Rest()
			else: 
				n = note.Note()
				n.pitch.midi = pitch[i,j]
			n.duration.quarterLength = duration[i,j]
			melody.append(n)
		# Convert and save stream to midi file
		mf = midi.translate.streamToMidiFile(melody)
		mf.open("melody_{}_".format(i, filename) + '.mid','wb')
		mf.write()
		mf.close()

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


def main():
	# path for pickled data files:
	# pkl_path = "../data/pkl/"
	# data_file_path = pkl_path + data_file + ".pkl"

	data_ohe, data = load_data(data_file="data_new", partition_file="partition", train_partition=0.8)
	pitch_decoded = one_hot_decoder(data_ohe["pitch"]["indices"], data_ohe["pitch"]["map_ind2feat"])
	duration_decoded = one_hot_decoder(data_ohe["duration"]["indices"], data_ohe["duration"]["map_ind2feat"])

	array2midi(pitch_decoded[0:2], duration_decoded[0:2], data["metadata"], filename="original")
	# print("Number of tunes in collection: {}".format(len(x_pitch)))
	# print(len(x_pitch[data_slices["jigs"]]))
	# #print(x_duration[data_slices["reels"]])
	
	# Printout Test of our one-hot-encoded (OHE) data structure 
	for s in range(2):
		print("\n{}. {}".format(s+1, data["metadata"][s]))
		for i in range(2):
			print("Note {}:".format(i+1))
			print("\tPitch:\n\t\tdecoded={}\toriginal={}".format(pitch_decoded[s,i], data["pitch"][s][i]))
			print("\tDuration:\n\t\tdecoded={}\toriginal={}".format(duration_decoded[s,i], data["duration"][s][i]))

if __name__ == '__main__':
	main()