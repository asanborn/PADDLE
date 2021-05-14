# PADDLE
#### (Predictor of Activation Domains using Deep Learning in Eukaryotes)

PADDLE is a deep convolutional neural network that predicts acidic transcriptional activation domains (ADs) from protein sequence. PADDLE can predict both the position and relative strength of acidic ADs. It was trained on high-throughput activation assay data in yeast (S. cerevisiae), but due to the high conservation of acidic AD function across eukaryotes, also predicts activation of human proteins in human cells. See [Sanborn et al. _eLife_ (2021)](https://https://elifesciences.org/articles/68068) for a full description of the experimental and computational methods.

Included here is:
- PADDLE model files, for loading using TensorFlow (models/)
- Sequence and predicted secondary structure of the Arg81 yeast transcription factor, for input to PADDLE (data/)
- Functions for running predictions using PADDLE (paddle.py)
- A Jupyter Notebook showing example predictions on wild-type and mutant protein sequences (PADDLE_predictions.ipynb)

PADDLE predictions for all nuclear proteins across multiple species has also been pre-computed and are available at [paddle.stanford.edu](http://paddle.stanford.edu).

PADDLE was developed by [Adrian Sanborn](http://www.adriansanborn.com) (Contact: a@adriansanborn.com) and Ben Yeh in the lab of Roger Kornberg at Stanford University.
