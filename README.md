# recurrent-neural-notes

Exam project in the [Deep Learning course][deep-learning] at DTU.

[deep-learning]: https://github.com/DeepLearningDTU/02456-deep-learning

A GRU-network model for next-step prediction of notes in the Nottingham folk-melody dataset. 

The model is also expanded to feeding the previous prediction from the output-network as input to the GRU-input-network, so as to be able to generate new sequences of notes from an initial one.

The final documentation of the project and results can be found [here](https://www.dropbox.com/s/i4fs83z07st24p3/report.pdf?dl=0). 

### Data pre-processing 
From ABC-format to Music21 objects to zero-padded one-hot encoded vectors for each note (pitch and duration) in each melody (list of lists of vectors --> numpy array X=[M, N, F]=[Melodies, Notes, Features]). Here all notes in all melodies are represented by a duration tensor, X<sub>d</sub>, with 14 features and a pitch tensor, X<sub>d</sub>, with 35 features. 

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Features.png" height="250">

### The next-step prediction GRU network models (with and without [orange] prediction input)

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Models.png" height="250">

### The learning curves for the 2 models with and without regularization.

Regularizing the GRU networks reduce overfitting, as seen by less span between training and validation accuracy curves. 

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/acc_learning_curves.png" width="800">

**Dropout:** By leaving out notes along the melodies, a lossy noise and therefore a completion task is introduced to the models, so during training the next-step prediction will rely more on the previous GRU activations h<sup>t-1</sup> and the horizontal connections will be enhanced to make up for the missing input.

**Prediction input**: By feeding in the previous prediction, a stronger loss signal will traverse across the horizontal connections and vanishing gradients can be avoided.   

### Histogram over pitch (left) and duration (right) usage in all melodies. 

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/models_pitch_freq_barplot.png" height="300"> <img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/models_duration_freq_barplot.png" height="300">

### Reconstructions of "The Fiddle Hill Jig" by Model 1 and 2. 

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Reconstructions_cut.png" width="600">

### GRU activations for two units in model 1.

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/model_1_activations_gru_26.png" width="400"> <img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/model_1_activations_gru_50.png" width="400">


