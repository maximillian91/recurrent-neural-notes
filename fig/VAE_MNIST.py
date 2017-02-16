import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.python.ops.nn import batch_normalization
from tensorflow.python.ops.nn import relu, sigmoid
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.distributions import Bernoulli, Normal, Poisson
#import ZeroInflatedPoisson
# from tensorflow.python.ops.control

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as gaussian

eps = 1e-10

def sample_layer(mean, log_var, scope='sample_layer'):
	with tf.variable_scope(scope):
		input_shape  = tf.shape(mean)
		batch_size = input_shape[0]
		num_latent = input_shape[1]
		eps = tf.random_normal((batch_size, num_latent), 0, 1, dtype=tf.float32)
		# Sample z = mu + sigma*epsilon
		return mean + tf.exp(0.5 * log_var) * eps
	
def dense_layer(inputs, num_outputs, is_training, scope, activation_fn=None, use_batch_norm=False, decay=0.999, center=True, scale=False):
	with tf.variable_scope(scope):
		outputs = fully_connected(inputs, num_outputs=num_outputs, activation_fn=None, scope='DENSE')
		if use_batch_norm:
			outputs = batch_norm(outputs, center=center, scale=scale, is_training=is_training, scope='BATCH_NORM')
		if activation_fn is not None:
			outputs = activation_fn(outputs)

		return outputs


def sum_of_squared_errors(p, t):
	return tf.reduce_sum(tf.square(p - t), axis=[1])


# computing cross entropy per sample
def categorical_cross_entropy(p, t, eps=0.0):
	return tf.reduce_sum(t * tf.log(p+eps), axis=[1])


def binary_cross_entropy(p, t, eps=0.0):
	return tf.reduce_sum(t * tf.log(p+eps) + (1-t) * tf.log(1-p+eps), axis=[1])

def kl_normal2_stdnormal(mean, log_var, eps=0.0):
	"""
    Compute analytically integrated KL-divergence between a diagonal covariance Gaussian and 
    a standard Gaussian.

    In the setting of the variational autoencoder, when a Gaussian prior and diagonal Gaussian 
    approximate posterior is used, this analytically integrated KL-divergence term yields a lower variance 
    estimate of the likelihood lower bound compared to computing the term by Monte Carlo approximation.

        .. math:: D_{KL}[q_{\phi}(z|x) || p_{\theta}(z)]

    See appendix B of [KINGMA]_ for details.

    Parameters
    ----------
    mean : Tensorflow tensor
        Mean of the diagonal covariance Gaussian.
    log_var : Tensorflow tensor
        Log variance of the diagonal covariance Gaussian.

    Returns
    -------
    Tensorflow tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).

	"""
	return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)


c = - 0.5 * math.log(2*math.pi)
def log_normal2(x, mean, log_var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Here variance is parameterized in the log domain, which ensures :math:`\sigma > 0`.

        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Tensorflow tensor
        Values at which to evaluate pdf.
    mean : Tensorflow tensor
        Mean of the Gaussian distribution.
    log_var : Tensorflow tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number used to avoid NaNs

    Returns
    -------
    Tensorflow tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    return tf.reduce_sum(c - log_var/2 - tf.square(x - mean) / (2 * tf.exp(log_var) + eps), axis=[1])


# reset graph
reset_default_graph()

# Load mnist
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# Synthetic Manifold latent representation
num_manifold_samples = 10
z_linspace = np.linspace(0.1, 0.9, num_manifold_samples)
# TODO: Ideally sample from the real p(z)
z_prob = gaussian.ppf(z_linspace)
z_samples = np.zeros((num_manifold_samples**2, 2))

i = 0
for a in z_prob:
	for b in z_prob:
		z_samples[i,0] = a
		z_samples[i,1] = b
		i += 1
z_samples = z_samples.astype('float32')


# Model specifications
N_epochs = 21
NUM_CLASSES = 2
NUM_FEAT = mnist.train.images[0].shape[0]
N_train = mnist.train.images.shape[0]
print(NUM_FEAT)
N_z = 2
BATCH_SIZE = 100
N_batches = N_train // BATCH_SIZE
reconstruction_distribution_name = 'poisson'
hidden_structure = [64, 32, 16]
use_batch_norm = True
plot_manifold = True


learning_rate = 1e-4
batch_norm_decay = 0.999

model_name = "VAE_" + reconstruction_distribution_name
if use_batch_norm:
	model_name += "_bn_" + str(batch_norm_decay) 
model_name += "_nz_" + str(N_z)
model_name += "_bs_" + str(BATCH_SIZE)
model_name += "_lr_" + str(learning_rate)
model_name += "_e_" + str(N_epochs)

# initialize placeholders, symbolics, with shape (batchsize, features)
x_in = tf.placeholder(tf.float32, [None, NUM_FEAT], 'x_in')
phase = tf.placeholder(tf.bool, [], 'phase')

l_enc = x_in
# Encoder - Recognition Model, q(z|x)
for i, hidden_size in enumerate(hidden_structure):
 	l_enc = dense_layer(inputs=l_enc, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=use_batch_norm, decay=batch_norm_decay,is_training=phase, scope='ENCODER{:d}'.format(i + 1))

l_mu_z = dense_layer(inputs=l_enc, num_outputs=N_z, activation_fn=None, use_batch_norm=False, is_training=phase, scope='ENCODER_MU_Z')
l_logvar_z = tf.clip_by_value(dense_layer(inputs=l_enc, num_outputs=N_z, activation_fn=None, use_batch_norm=False, is_training=phase, scope='ENCODER_LOGVAR_Z'), -10, 10)

# Stochastic layer
## Sample latent variable: z = mu + sigma*epsilon
l_z = sample_layer(l_mu_z, l_logvar_z, 'SAMPLE_LAYER')

# Decoder - Generative model, p(x|z)
l_dec = l_z
for i, hidden_size in enumerate(reversed(hidden_structure)):
	l_dec = dense_layer(inputs=l_dec, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=use_batch_norm, decay=batch_norm_decay, is_training=phase, scope='DECODER{:d}'.format(i + 1))

# Reconstruction Distribution Parameterization
if reconstruction_distribution_name == 'bernoulli':
	l_dec_out_p = dense_layer(inputs=l_dec, num_outputs=NUM_FEAT, activation_fn=sigmoid, use_batch_norm=False, is_training=phase, scale=True, scope='DECODER_BERNOULLI_P')
	recon_dist = Bernoulli(p = l_dec_out_p)

elif reconstruction_distribution_name == 'normal':
	l_dec_out_mu = dense_layer(inputs=l_dec, num_outputs=NUM_FEAT, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_NORMAL_MU')
	l_dec_out_log_sigma = dense_layer(inputs=l_dec, num_outputs=NUM_FEAT, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_NORMAL_LOG_SIGMA')
	recon_dist = Normal(mu=l_dec_out_mu, 
		sigma=tf.exp(tf.clip_by_value(l_dec_out_log_sigma, -3, 3)))

elif reconstruction_distribution_name == 'poisson':
	l_dec_out_log_lambda = dense_layer(inputs=l_dec, num_outputs=NUM_FEAT, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_POISSON_LOG_LAMBDA')
	recon_dist = Poisson(lam=tf.exp(tf.clip_by_value(l_dec_out_log_lambda, -10, 10)))

# Loss
# Reconstruction error. (all log(p) are in [-\infty, 0]). 
log_px_given_z = tf.reduce_sum(recon_dist.log_prob(x_in), axis = 1)
# Regularization: Kulback-Leibler divergence between approximate posterior, q(z|x), and isotropic gauss prior p(z)=N(z,mu,sigma*I).
KL_qp = kl_normal2_stdnormal(l_mu_z, l_logvar_z, eps=eps)

# Averaging over samples.  
loss = tf.reduce_mean(log_px_given_z - KL_qp, name="ELBO")


# Make sure that the Updates of the moving_averages in batch_norm layers
# are performed before the train_step.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(update_ops)
if update_ops:
	updates = tf.group(*update_ops)
	with tf.control_dependencies([updates]):
		# Optimizer and training objective of negative loss
		train_step = tf.train.AdamOptimizer(1e-4).minimize(-loss)
else: 
	train_step = tf.train.AdamOptimizer(1e-4).minimize(-loss)	



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	params = tf.trainable_variables()

	for param in params:
		print(param.name, param.get_shape())
	# # test the forward pass
	# x_dummy = np.zeros(shape=(BATCH_SIZE, NUM_FEAT))

	# # test the forward pass
	# feed_dict_dummy = {x_in: x_dummy, is_training: False}
	# res_forward_pass = sess.run(fetches=recon_dist.mean(), feed_dict=feed_dict_dummy)
	# print("x_mean shape", res_forward_pass.shape)


	# Train
	train_losses, valid_losses = [], []
	feed_dict_train = {'x_in:0': mnist.train.images, 'phase:0': False}
	feed_dict_valid = {'x_in:0': mnist.validation.images, 'phase:0': False}
	for epoch in range(N_epochs):
		for i in range(N_batches):
			batch = mnist.train.next_batch(BATCH_SIZE)
			epoch = mnist.train.epochs_completed
			_, batch_loss = sess.run([train_step, loss], feed_dict={'x_in:0': batch[0], 'phase:0': True})
		
		train_loss = sess.run(loss, feed_dict=feed_dict_train)
		valid_loss = sess.run(loss, feed_dict=feed_dict_valid)


		print("Epoch %d: ELBO: %g (Train), %g (Valid)"%(epoch+1, train_loss, valid_loss))

		#logpxz = sess.run(log_p_x_z, feed_dict={'x_in:0': batch[0]})
		#print(logpxz.shape)
		train_losses += [train_loss]
		valid_losses += [valid_loss]


	test_loss = sess.run(loss, feed_dict={'x_in:0': mnist.test.images, 'phase:0': False})
	print("Total Epochs %d: ELBO: %g (Train), %g (Valid), %g (Test)"%(epoch, train_loss, valid_loss, test_loss))


	# Plot Learning Curves
	plt.title('Loss (ELBO)')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.xlabel('Epochs Trained'), plt.ylabel('ELBO')
	plt.plot(train_losses, color="black")
	plt.plot(valid_losses, color="grey")
	#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.grid('on')
	plt.savefig(model_name + "_LearningCurves.png")


	# Plot manifold samples
	if plot_manifold:
		feed_dict_sample = {l_z: z_samples, 'phase:0': False}
		x_samples = sess.run(recon_dist.mean(), feed_dict=feed_dict_sample)
		plt.cla()	
		plt.title('Manifold')
		#plt.axis('off')
		idx = 0
		canvas = np.zeros((28*num_manifold_samples, num_manifold_samples*28))
		for i in range(10):
			for j in range(10):
				canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_samples[idx].reshape((28, 28))
				idx += 1
		plt.imshow(canvas, cmap='gray')
		plt.savefig(model_name + "_Manifold.png")








# def log_poisson(x, log_lambda, eps = 0.0):
    
#     x = tf.clip_by_value(x, eps, ) #T.clip(x, eps, x)
    
#     lambda_ = T.exp(log_lambda)
#     lambda_ = T.clip(lambda_, eps, lambda_)
    
#     y = x * log_lambda - lambda_ - T.gammaln(x + 1)
    
#     return y

# tf.nn.log_poisson_loss(log_input, targets, compute_full_loss=False, name=None)