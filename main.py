"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--target`: path to target dir
* `--output : path to output dir
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

"""
Inspired from Yann Soulard github and paper. The Dataset has been given 
by Yann Soulard himself during Sequential Data Analysis, during my 
last M2 year. 
"""

import os
import h5py
import utils 
import string
import argparse
import datetime
import numpy as np
import tensorflow as tf 

from model import CTCModel
from generator import DataGenerator

NB_FILTERS = 50

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--output", type=str, default='output/')
    parser.add_argument("--target", type=str, default='target/')


    args = parser.parse_args()
    print("Parsers OK", args)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.target, exist_ok=True)

    dtgen = DataGenerator()
    print("Generator OK")



    network = CTCModel(NB_FILTERS, dtgen.max_len, dtgen.nb_labels + 1)
    print("Network OK")
    network.compile(0.001)
    network.summary()
    callbacks = network.get_callbacks(logdir=args.output, checkpoint=args.target + "checkpoints_", verbose=1)

    start_time = datetime.datetime.now()

    h = network.fit(x=dtgen.next_train_batch(),
                 epochs=args.epochs,
                 steps_per_epoch=dtgen.steps['train'],
                 validation_data=dtgen.next_valid_batch(),
                 validation_steps=dtgen.steps['valid'],
                 callbacks=callbacks,
                 shuffle=True,
                 verbose=1)

    total_time = datetime.datetime.now() - start_time
    
    loss = h.history['loss']
    val_loss = h.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_i = val_loss.index(min_val_loss)

    time_epoch = (total_time / len(loss))
    print("total time : ", total_time)
    
    total_item = (dtgen.size['train'] + dtgen.size['valid'])
    print(total_item)