from data import load_data
from aux import models_path
import cPickle as pickle
import os.path
from os import listdir

import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks', palette='Set2')


####### RNN Model for folk music composition ########

print(os.getcwd())

# Defining model path
#model_numbering = [int(directory[-1]) for directory in listdir("../models/") if file[:6]=="model_"]

#if isempty(model_numbering):
#	new_model_number = 0
#else:
#	new_model_number = max(model_numbering) + 1

new_model_number = 1

model_dir = "model_" + str(new_model_number)

fig_path = models_path("model_1/fig")
fig_path += "/"

# Importing data
data, _ = load_data(data_file="data_with_eos", partition_file="partition", train_partition=0.8)

data_pitch_ohe = data["pitch"]["encoded"]
data_pitch = data["pitch"]["indices"]
data_duration_ohe = data["duration"]["encoded"]
data_duration = data["duration"]["indices"]


# Setting model parameters
data_pitch_shape = np.shape(data_pitch_ohe)
N_total = data_pitch_shape[0]
MAX_SEQ_LEN = data_pitch_shape[1]-1
NUM_FEATURES_pitch = data_pitch_shape[2]

data_duration_shape = np.shape(data_duration_ohe)
NUM_FEATURES_duration = data_duration_shape[2]

NUM_FEATURES_total = NUM_FEATURES_duration + NUM_FEATURES_pitch

BATCH_SIZE = 10
NUM_UNITS_ENC = 25
NUM_UNITS_DEC = 25


#symbolic theano variables. Note that we are using imatrix for X since it goes into the embedding layer
x_pitch_sym  = T.ltensor3('x_pitch_sym')
x_duration_sym = T.ltensor3('x_duration_sym')

y_pitch_sym = T.ltensor3('y_pitch_sym')
y_duration_sym = T.ltensor3('y_duration_sym')

mask_sym = T.matrix('mask_sym')

#dummy data to test implementation - We advise to check the output-dimensions of all layers.
#One way to do this in lasagne/theano is to forward pass some data through the model and 
#check the output dimensions of these.
#Create some random testdata for pitch input
X_pitch = np.zeros((BATCH_SIZE, MAX_SEQ_LEN, NUM_FEATURES_pitch)).astype('int32')
for i in range(BATCH_SIZE):
	for j in range(MAX_SEQ_LEN):
		k = (i*MAX_SEQ_LEN + j) % NUM_FEATURES_pitch
		X_pitch[i,j,k] = 1

X_pitch = X_pitch.astype('int32')
print(np.shape(X_pitch))
X_mask = np.ones((BATCH_SIZE,MAX_SEQ_LEN)).astype('float32')
print(np.shape(X_mask))

X_duration = np.zeros((BATCH_SIZE, MAX_SEQ_LEN, NUM_FEATURES_duration)).astype('int32')
for i in range(BATCH_SIZE):
	for j in range(MAX_SEQ_LEN):
		k = (i*MAX_SEQ_LEN + j) % NUM_FEATURES_duration
		X_duration[i,j,k] = 1

X_duration = X_duration.astype('int32')
print(np.shape(X_duration))


##### ENCODER START #####
# Skip this One-hot-encoding step as the input data is already OHE like: input_pitch = (BATCH_SIZE, MAX_SEQ_LEN, NUM_PITCHES) and input_duration = (BATCH_SIZE, MAX_SEQ_LEN, NUM_DURATIONS) 
#l_in = lasagne.layers.InputLayer((None, None, NUM_FEATURES))

l_in_pitch = lasagne.layers.InputLayer((None, None, NUM_FEATURES_pitch), name="l_in_pitch")
l_in_duration = lasagne.layers.InputLayer((None, None, NUM_FEATURES_duration), name="l_in_duration")

#l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS, W=np.eye(NUM_INPUTS,dtype='float32'), name='Embedding')
#Here we'll remove the trainable parameters from the embeding layer to constrain 
#it to a simple "one-hot-encoding". You can experiment with removing this line
#l_emb.params[l_emb.W].remove('trainable') 
#forward pass some data throug the inputlayer-embedding layer and print the output shape
print l_in_pitch.name, ":", lasagne.layers.get_output(l_in_pitch, inputs={l_in_pitch: x_pitch_sym}).eval({x_pitch_sym: X_pitch}).shape
print l_in_duration.name, ":", lasagne.layers.get_output(l_in_duration, inputs={l_in_duration: x_duration_sym}).eval({x_duration_sym: X_duration}).shape


l_in_merge = lasagne.layers.ConcatLayer([l_in_pitch, l_in_duration], axis=2, name="l_in_merge")
print l_in_merge.name, ":", lasagne.layers.get_output(l_in_merge, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration}).shape


l_mask_enc = lasagne.layers.InputLayer((None, MAX_SEQ_LEN), name="l_mask_enc")

l_enc = lasagne.layers.GRULayer(l_in_merge, num_units=NUM_UNITS_ENC, name='GRUEncoder', mask_input=l_mask_enc)
print l_enc.name, ":", lasagne.layers.get_output(l_enc, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape
# Don't slice, but keep the full hidden state enc
##### END OF ENCODER ######


##### START OF DECODER #####
l_dec = lasagne.layers.GRULayer(l_enc, num_units=NUM_UNITS_DEC, name='GRUDecoder')
print l_dec.name, ":", lasagne.layers.get_output(l_dec, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape


# We need to do some reshape voodo to connect a softmax layer to the decoder.
# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples 
# In short this line changes the shape from 
# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units). 
# We need to do this since the softmax is applied to the last dimension and we want to 
# softmax the output at each position individually
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]), name="l_reshape")
print l_reshape.name, ":", lasagne.layers.get_output(l_reshape, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape

l_softmax_pitch = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_FEATURES_pitch, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_pitch')
print l_softmax_pitch.name, ": ", lasagne.layers.get_output(l_softmax_pitch, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape

l_softmax_duration = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_FEATURES_duration, nonlinearity=lasagne.nonlinearities.softmax, name='SoftmaxOutput_duration')
print l_softmax_duration.name, ": ", lasagne.layers.get_output(l_softmax_duration, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape

# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing 
#us to use different batch sizes in the model.
l_out_pitch = lasagne.layers.ReshapeLayer(l_softmax_pitch, (-1, MAX_SEQ_LEN, NUM_FEATURES_pitch), name="l_out_pitch")
print l_out_pitch.name, ":", lasagne.layers.get_output(l_out_pitch, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape

l_out_duration = lasagne.layers.ReshapeLayer(l_softmax_duration, (-1, MAX_SEQ_LEN, NUM_FEATURES_duration), name="l_out_duration")
print l_out_duration.name, ":", lasagne.layers.get_output(l_out_duration, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}).eval({x_pitch_sym: X_pitch, x_duration_sym: X_duration, mask_sym: X_mask}).shape

###END OF DECODER######

output_pitch = lasagne.layers.get_output(l_out_pitch, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}, deterministic=False)
output_duration = lasagne.layers.get_output(l_out_duration, inputs={l_in_pitch: x_pitch_sym, l_in_duration: x_duration_sym, l_mask_enc: mask_sym}, deterministic=False)
#output_decoder_eval = lasagne.layers.get_output(l_out, inputs={l_in_pitch: x_pitch_sym, x_duration_sym,l_mask_enc: mask_sym}, deterministic=True)


def evaluate(output, target, num_features, mask):
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

cost_pitch, acc_pitch = evaluate(output_pitch, y_pitch_sym, NUM_FEATURES_pitch, mask_sym)
cost_duration, acc_duration = evaluate(output_duration, y_duration_sym, NUM_FEATURES_duration, mask_sym)
total_cost = cost_pitch + cost_duration

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params([l_out_pitch, l_out_duration], trainable=True)

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
f_train = theano.function([x_pitch_sym, y_pitch_sym, x_duration_sym, y_duration_sym, mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration], updates=updates)
#since we don't have any stochasticity in the network we will just use the training graph without any updates given
f_eval = theano.function([x_pitch_sym, y_pitch_sym, x_duration_sym, y_duration_sym, mask_sym], [cost_pitch, acc_pitch, output_pitch, cost_duration, acc_duration, output_duration])


### TRAINING ###
#Collect data
train_idx = data["train_idx"]
valid_idx = data["valid_idx"]
test_idx = data["test_idx"]
input_mask = data["mask"]

# Inputs
x_pitch = data_pitch_ohe[:,:-1] 
x_pitch_valid = x_pitch[valid_idx]
x_pitch_train = x_pitch[train_idx]
x_pitch_test = x_pitch[test_idx]

# Targets
y_pitch = data_pitch_ohe[:,1:]
y_pitch_valid = y_pitch[valid_idx]
y_pitch_train = y_pitch[train_idx]
y_pitch_test = y_pitch[test_idx]

# Inputs
x_duration = data_duration_ohe[:,:-1] 
x_duration_valid = x_duration[valid_idx]
x_duration_train = x_duration[train_idx]
x_duration_test = x_duration[test_idx]

# Targets
y_duration = data_duration_ohe[:,1:]
y_duration_valid = y_duration[valid_idx]
y_duration_train = y_duration[train_idx]
y_duration_test = y_duration[test_idx]

# Masks
mask_train = input_mask[train_idx][:,:-1] 
mask_valid =  input_mask[valid_idx][:,:-1] 
mask_test =  input_mask[test_idx][:,:-1] 


print "x_pitch_train", x_pitch_train.shape
print "y_pitch_train", y_pitch_train.shape
print "mask_train", mask_train.shape
print "x_pitch_valid", x_pitch_valid.shape
print "y_pitch_valid", y_pitch_valid.shape
print "mask_valid", mask_valid.shape

print "x_duration_train", x_duration_train.shape
print "y_duration_train", y_duration_train.shape
#print "Yval", Yval.shape

N_epochs = 100

print("Training model.")

cost_train_pitch = []
acc_train_pitch = []
cost_valid_pitch = []
acc_valid_pitch = []
cost_train_duration = []
acc_train_duration = []
cost_valid_duration = []
acc_valid_duration = []

N_train = x_pitch_train.shape[0]

header_string = "Cost:\tPitch\tDuration| Acc:\tPitch\tDuration"
valid_string = ""
for epoch in range(N_epochs):
	epoch_cost = 0
	shuffled_indices = np.random.permutation(N_train)
	for i in range(0, N_train, BATCH_SIZE):
		# Collect random batch 
		subset = shuffled_indices[i:(i + BATCH_SIZE)]
		x_pitch_batch = x_pitch_train[subset]
		y_pitch_batch = y_pitch_train[subset]
		x_duration_batch = x_duration_train[subset]
		y_duration_batch = y_duration_train[subset]
		mask_batch = mask_train[subset]
		# Train for batch and collect cost, accuracy and output
		batch_cost_pitch, batch_acc_pitch, batch_output_pitch, batch_cost_duration, batch_acc_duration, batch_output_duration = f_train(x_pitch_batch, y_pitch_batch, x_duration_batch, y_duration_batch, mask_batch)
		# epoch_cost += batch_cost
	train_cost_pitch, train_acc_pitch, train_output_pitch, train_cost_duration, train_acc_duration, train_output_duration = f_eval(x_pitch_train, y_pitch_train, x_duration_train, y_duration_train, mask_train)
	train_string = "Train: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(train_cost_pitch), float(train_cost_duration), float(train_acc_pitch), float(train_acc_duration))

	cost_train_pitch += [train_cost_pitch]
	acc_train_pitch += [train_acc_pitch]
	cost_train_duration += [train_cost_duration]
	acc_train_duration += [train_acc_duration]

	if x_pitch_valid is not None:
		valid_cost_pitch, valid_acc_pitch, valid_output_pitch, valid_cost_duration, valid_acc_duration, valid_output_duration = f_eval(x_pitch_valid, y_pitch_valid, x_duration_valid, y_duration_valid, mask_valid)
		cost_valid_pitch += [valid_cost_pitch]
		acc_valid_pitch += [valid_acc_pitch]
		cost_valid_duration += [valid_cost_duration]
		acc_valid_duration += [valid_acc_duration]
		valid_string = "Valid: \t\t\t{:.4g}\t{:.4g}\t|\t\t{:.4g}\t{:.4g}".format(float(valid_cost_pitch), float(valid_cost_duration), float(valid_acc_pitch), float(valid_acc_duration))
	
	epoch_string = "\nEpoch {:2d}: {}\n{}\n{}".format(epoch + 1, header_string, train_string, valid_string)
	print(epoch_string)


# Inspect model predictions for validation set examples:
number_of_test_examples = 3
test_slice = slice(number_of_test_examples)
test_cost_pitch, test_acc_pitch, test_output_pitch, test_cost_duration, test_acc_duration, test_output_duration = f_eval(x_pitch_valid[test_slice,:,:], y_pitch_valid[test_slice,:,:], x_duration_valid[test_slice,:,:], y_duration_valid[test_slice,:,:], mask_valid[test_slice,:])

max_prob_pitch = np.argmax(test_output_pitch,axis=2)
max_prob_duration = np.argmax(test_output_duration,axis=2)
test_ind = np.nonzero(mask_valid[test_slice])

print("Inspect the first {} melodies:".format(number_of_test_examples))
for i in range(number_of_test_examples):
	print("Pitch targets and prediction")
	print(max_prob_pitch[i])
	print(data_pitch[i])

	print("Duration targets and prediction")
	print(max_prob_duration[i])
	print(data_duration[i])

# Accuracy plots
# Pitch
plt.figure()
plt.plot(acc_train_pitch)
plt.plot(acc_valid_pitch)
plt.ylabel('Validation Accuracy (Pitch)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "acc_pitch.png")

plt.figure()
plt.scatter(acc_train_pitch, acc_valid_pitch)
eq_line = range(max(acc_train_pitch, acc_valid_pitch))
plt.plot(eq_line)
plt.ylabel('Accuracy Trajectory (Pitch)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "acc_traj_pitch.png")

## Duration 
plt.figure()
plt.plot(acc_train_duration)
plt.plot(acc_valid_duration)
plt.ylabel('Validation Accuracy (Duration)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "acc_duration.png")

plt.figure()
plt.scatter(acc_train_duration, acc_valid_duration)
eq_line = range(max(acc_train_duration, acc_valid_duration))
plt.plot(eq_line)
plt.ylabel('Validation Accuracy', fontsize=15)
plt.xlabel('Training Accuracy', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "acc_traj_duration.png")

# Cost plots
## Accuracy
plt.figure()
plt.plot(cost_train_pitch)
plt.plot(cost_valid_pitch)
plt.ylabel('Validation Accuracy (Pitch)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "cost_pitch.png")

plt.figure()
plt.scatter(cost_train_pitch, cost_valid_pitch)
eq_line = range(max(cost_train_pitch, cost_valid_pitch))
plt.plot(eq_line)
plt.ylabel('Accuracy Trajectory (Pitch)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "cost_traj_pitch.png")

## Duration
plt.figure()
plt.plot(cost_train_duration)
plt.plot(cost_valid_duration)
plt.ylabel('Validation Accuracy (Duration)', fontsize=15)
plt.xlabel('Epoch #', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "cost_duration.png")

plt.figure()
plt.scatter(cost_train_duration, cost_valid_duration)
eq_line = range(max(cost_train_duration, cost_valid_duration))
plt.plot(eq_line)
plt.ylabel('Validation Accuracy', fontsize=15)
plt.xlabel('Training Accuracy', fontsize=15)
plt.title('', fontsize=20)
plt.grid('on')
plt.savefig(fig_path + "cost_traj_duration.png")

