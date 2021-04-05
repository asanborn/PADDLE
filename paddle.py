#!/bin/usr/env python

import os
import numpy as np
import pandas as pd
import tensorflow as tf # tested on version 2.2.0

tf.get_logger().setLevel('ERROR') # Ignore warning messages


def read_ss2(file):
    """
    Read secondary structure scores produced by PSIPRED V4.0 from the .ss2 file.
    Inputs:
        - file: Filename of the .ss2 file output by PSIPRED V4.0
    Returns:
        - prot:  Protein sequence (string)
        - helix: List of predicted helix tendencies (values between 0 and 1)
        - coil:  List of predicted coil tendencies (values between 0 and 1)
    """
    
    f = open(file, 'r')
    helix, coil = [], [] # Ignores beta sheet score since the three scores sum to 1
    prot = ''
    
    for line in f:
        if '#' in line or len(line.split()) < 4:
            continue
        
        tokens = line.strip().split()
        assert len(tokens) == 6
        prot += tokens[1]
        helix.append(float(tokens[4]))
        coil.append(float(tokens[3]))
    f.close()
    
    return prot, helix, coil


def read_iupred(file):
    """
    Read disorder predictions produced by IUPRED2 from the .dis file
    Inputs:
        - file: Filename of the .dis file output by IUPRED2
    Returns:
        - prot: Protein sequence (string)
        - dis:  List of predicted disorder (values between 0 and 1)
    """
    
    f = open(file, 'r')
    dis = []
    prot = ''
    
    for line in f:
        if '#' in line:
            continue
        tokens = line.split()
        assert len(tokens) == 3
        
        prot += tokens[1]
        dis.append(float(tokens[2]))
        
    f.close()
    return prot, dis


def encode_onehot(prot, AAs = list('ACDEFGHIKLMNPQRSTVWY')):
    """
    Encode the protein sequence as a one-hot vector.
    Inputs
        - prot: The protein sequence as a list of amino acids.
                When feeding into PADDLE, should be 53aa long.
        - AAs:  The list of amino acids.
    Returns
        - out:  The one-hot encoding of the protein. If the input is 53aa long
                then this will be a numpy array of size (53,20). Any chars
                outside of the standard 20 amino acids will be ignored.
    """
    
    out = np.zeros((len(prot), len(AAs)))
    for i, aa in enumerate(prot):
        try:
            out[i, AAs.index(aa)] = 1
        # If amino acid is not found, leave as a vector of all 0s
        except ValueError:
            pass
    return out


def load_models(model_name, model_dir='models', splits=range(10)):
    """
    Load all 10 PADDLE TensorFlow models. (PADDLE consists of 10 CNNs of the
    same architecture trained on 10 different splits of the data.)
    Inputs:
        - model_name: Should be 'PADDLE' or 'PADDLE_noSS'
        - model_dir:  The directory containing the model files
        - splits:     The list of model splits to load.
    Outputs:
        - models:     A list of all 10 TensorFlow models
    """
    
    models = []
    for i in splits:
        print(f'Loading model split {i+1} of {len(splits)}')
        model_file = os.path.join(model_dir, model_name + f'_split{i}.model')
        models.append(tf.keras.models.load_model(model_file))
    return models


def score2act(score):
    """
    Convert the Z-score predicted by PADDLE to a fold-activation value.
    Only useful as a rough reference for activation in S. cerevisiae, as the
    fold-activation will vary between different experimental conditions.
    """
    return 1.5031611623938073**score


class PADDLE:
    """
    A class to run PADDLE predictions. This network requires predicted secondary
    structure (from PSIPRED) and predicted disorder (from IUPRED2, in both the
    short and long modes) as input in addition to the protein sequence. This
    model is the most accurate and should be used for predicted ADs in wild-type
    proteins.
    Note: when predicting across a large number of sequences, PSIPRED can be run
    without using BLAST, this speeds up secondary structure prediction. 
    """
    
    def __init__(self):
        self.models = load_models('PADDLE')
        self.offset = 2.4845441766212284
        
        
    def predict_protein(self, prot, helix, coil, dis_short, dis_long, smooth=9):
        """
        Generate predictions for all 53aa tiles across a protein. By default,
        the predicted Z-scores are smoothed by taking a 9aa sliding window average.
        Inputs:
            prot:      Protein amino acid sequence (string)
            helix:     Helix scores predicted by PSIPRED (list)
            coil:      Coil scores predicted by PSIPRED (list)
            dis_short: Disorder scores predicted by IUPRED2 in 'short' mode (list)
            dis_long:  Disorder scores predicted by IUPRED2 in 'long' mode (list)
            smooth:    Number of amino acids for the sliding window average.
                       Set to <=1 to turn off smoothing. (int)
        Outputs:
            predictions: A Numpy array of predicted Z-scores for each 53aa tile
                         in the protein; this will have length = len(prot)-52.
        """
        
        assert len(prot) >= 53
        
        # Stack the protein one-hot encoding with the predicted structure scores
        annotations = np.stack([helix, coil, dis_short, dis_long]).T
        encoding = [np.hstack([encode_onehot(prot[i:i+53]), annotations[i:i+53]])
                    for i in range(len(prot)-52)]
        encoding = np.array(encoding)
        
        # Run predictions
        predictions = np.hstack([model.predict(encoding) for model in self.models])
        predictions = np.mean(predictions, axis=-1) + self.offset
        
        # Smooth predictions with a sliding mean with window size `smooth`
        if smooth > 1:
            predictions = [np.mean(predictions[max(0, int(n-(smooth-1)/2)) : 
                                               int(n+(smooth+1)/2)])
                           for n in range(len(predictions))]
            
        return np.array(predictions)
        
        
class PADDLE_noSS:
    """
    A class to run PADDLE-noSS predictions. This network does not need predicted
    structure scores from PSIPRED and IUPRED2 and can therefore run very fast, and
    is nearly as accurate as PADDLE (with structure input). This model can be used
    when running predictions on a large number of mutant sequences, for example.
    """
    
    def __init__(self):
        self.models = load_models('PADDLE_noSS')
        self.offset = 2.4845441766212284
        
        
    def predict(self, seqs):
        """
        Predict activation for a list of 53aa sequences. Can also accept a single
        53aa sequence. Note: 1000 predictions run all together will complete much
        more quickly than 1000 individual predictions.
        Inputs:
            seqs: A list of 53aa-long protein sequences to predict on. (list)
                  Can also accept a single 53aa sequence. (string)
        Outputs:
            predictions: Array of predicted Z-scores for each sequence. (array)
                         If a sequence was input, will return one value. (float)
        """
        
        if type(seqs) is str:
            seqs = [seqs]
            
        for seq in seqs:
            assert len(seq) == 53 # each sequence should be 53aa long
            
        # Encode protein sequences as one-hot and stack them into one array
        encoding = np.array([encode_onehot(seq) for seq in seqs])
        # Predict across the 10 models and take the mean value.
        predictions = np.hstack([model.predict(encoding) for model in self.models])
        predictions = np.mean(predictions, axis=-1) + self.offset
        
        # Return a float if only one sequence was given; otherwise return list
        if len(seqs) == 1:
            return predictions[0]
        else:
            return predictions
    
    
    def predict_subsequences(self, seqs, bg_prots):
        """
        Predict activation Z-scores for a list of sequences that can be shorter
        than 53aa long. Sequences shorter than 53aa are embedded at the center of
        53aa neutral background sequences (bg_prots) and the mean Z-score
        across all such embeddings is averaged. Sequences can also contain 'X',
        in which case the amino acid from the background sequence will be kept.
        Like predict(), can predict on a list of sequences or a single sequence.
        Inputs:
            seqs:     A list of 53aa-long protein sequences to predict on. (list)
                      Can also accept a single 53aa sequence. (string)
            bg_prots: A list of background sequences, each 53aa long. Random
                      sequences consisting of AGSTNQV amino acids work well. To
                      reduce their influence on the predicted values, 10 - 100 
                      such sequences should be provided. (list)
        Outputs:
            predictions: Array of predicted Z-scores for each sequence. (array)
                         If a sequence was input, will return one value. (float)
        """
        
        if type(seqs) is str:
            seqs = [seqs]
            
        for seq in seqs:
            assert len(seq) <= 53
           
        # Embed each protein sequence at the center of each background sequence
        seq_embed = []
        for seq in seqs:
            L = len(seq)
            bg_pos = int((53 - L)/2) # embedding start position
            
            for bg in bg_prots:
                # make the embedded sequence
                new_seq = list(bg[:bg_pos] + seq + bg[bg_pos+L:])    
                
                # handle any 'X' values
                if 'X' in new_seq:
                    for i in range(len(new_seq)):
                        if new_seq[i] == 'X':
                            new_seq[i] = bg[i]
                            
                seq_embed.append(''.join(new_seq))
        
        # Predict all values and average across background sequences to 
        # minimize their influence.
        preds = self.predict(seq_embed)
        mean_preds = np.mean(preds.reshape(len(seqs),len(bg_prots)), axis=1)
        assert len(mean_preds) == len(seqs)
        
        # Return a float if only one sequence was given; otherwise return list
        if len(seqs) == 1:
            return mean_preds[0]
        else:
            return mean_preds