# recurrent-neural-notes

Exam project in the [Deep Learning course][deep-learning] at DTU.

[deep-learning]: https://github.com/DeepLearningDTU/02456-deep-learning

A GRU-network model for next-step prediction of notes in the Nottingham folk-melody dataset. 

The model is also expanded to feeding the previous prediction from the output-network as input to the GRU-input-network, so as to be able to generate new sequences of notes from an initial one.

The final documentation of the project and results can be found [here](https://www.dropbox.com/s/i4fs83z07st24p3/report.pdf?dl=0). 

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Features.png" width="200"> <img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Models.png" width="600">

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/acc_learning_curves.png" width="800">

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/models_pitch_freq_barplot.png" width="600"> <img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/models_duration_freq_barplot.png" width="200">

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/Reconstructions_cut.png" width="600">

<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/model_1_activations_gru_26.png" width="600"> 
<img src="https://github.com/maximillian91/recurrent-neural-notes/blob/master/fig/model_1_activations_gru_50.png" width="600">


