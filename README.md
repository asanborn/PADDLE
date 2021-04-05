# PADDLE
#### (Predictor of Activation Domains using Deep Learning in Eukaryotes)

PADDLE is a tool for predicting transcription factor activation domain regions from protein sequence. PADDLE is a deep convolutional neural network trained on high-throughput activation assay data in yeast and can accurately predict the position and strength of activation domains. It has been experimentally validated to predict activation of human protein regions in human cells. See [Sanborn et al. Biorxiv (2020)](https://www.biorxiv.org/content/10.1101/2020.12.18.423551v1) for more details. 

Included here is:
- PADDLE model files, for loading using TensorFlow (models/)
- Sequence and predicted secondary structure of the Arg81 yeast transcription factor, for input to PADDLE (data/)
- Functions for running predictions using PADDLE (paddle.py)
- A Jupyter Notebook showing example predictions on wild-type and mutant protein sequences (PADDLE_predictions.ipynb)

PADDLE predictions for all nuclear proteins across multiple species has also been pre-computed and are available at [paddle.stanford.edu](http://paddle.stanford.edu).

PADDLE was developed by [Adrian Sanborn](http://www.adriansanborn.com) (Contact: a@adriansanborn.com).
