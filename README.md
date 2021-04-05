# PADDLE
PADDLE (Predictor of Activation Domains using Deep Learning in Eukaryotes) is a tool for predicting transcription factor activation domain regions from protein sequence. PADDLE is a deep convolutional neural network trained on high-throughput activation assay data in yeast and can accurately predict the position and strength of activation domains. It has been experimentally validated to predict activation of human protein regions in human cells. See Sanborn et al. Biorxiv (2020) for more details. (https://www.biorxiv.org/content/10.1101/2020.12.18.423551v1)

Included here is:
- PADDLE model files, for loading using TensorFlow
- Functions for running predictions using PADDLE
- A Jupyter Notebook showing example predictions based on the Arg81 yeast transcription factor

PADDLE predictions for all nuclear proteins across multiple species has also been pre-computed and are available at http://paddle.stanford.edu 

PADDLE was developed by Adrian Sanborn (Contact: a@adriansanborn.com)
