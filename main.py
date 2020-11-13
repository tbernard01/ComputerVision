"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import os
# import cv2
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
    parser.add_argument("--output", type=str, default='output')
    parser.add_argument("--target", type=str, default='target')


    args = parser.parse_args()
    print("Parsers OK")

    # raw_path = os.path.join("..", "raw", args.source)
    # source_path = os.path.join("..", "data", f"{args.source}.hdf5")
    # output_path = os.path.join("..", "output", args.source, args.arch)
    # target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.target, exist_ok=True)

    dtgen = DataGenerator()
    print("Generator OK")



    network = CTCModel(NB_FILTERS, dtgen.max_len, dtgen.nb_labels + 1)
    print("Network OK")
    network.compile(0.001)
    network.summary()
    callbacks = network.get_callbacks(logdir=args.output, checkpoint=args.target, verbose=1)

    start_time = datetime.datetime.now()

    h = network.fit(x=dtgen.next_train_batch(),
                 epochs=args.epochs,
                 steps_per_epoch=dtgen.steps['train'],
                 validation_data=dtgen.next_valid_batch(),
                 validation_steps=dtgen.steps['valid'],
                 callbacks=callbacks,
                 shuffle=True,
                 verbose=1)

        #     total_time = datetime.datetime.now() - start_time

        #     loss = h.history['loss']
        #     val_loss = h.history['val_loss']

        #     min_val_loss = min(val_loss)
        #     min_val_loss_i = val_loss.index(min_val_loss)

        #     time_epoch = (total_time / len(loss))
        #     total_item = (dtgen.size['train'] + dtgen.size['valid'])

        #     t_corpus = "\n".join([
        #         f"Total train images:      {dtgen.size['train']}",
        #         f"Total validation images: {dtgen.size['valid']}",
        #         f"Batch:                   {dtgen.batch_size}\n",
        #         f"Total time:              {total_time}",
        #         f"Time per epoch:          {time_epoch}",
        #         f"Time per item:           {time_epoch / total_item}\n",
        #         f"Total epochs:            {len(loss)}",
        #         f"Best epoch               {min_val_loss_i + 1}\n",
        #         f"Training loss:           {loss[min_val_loss_i]:.8f}",
        #         f"Validation loss:         {min_val_loss:.8f}"
        #     ])

        #     with open(os.path.join(output_path, "train.txt"), "w") as lg:
        #         lg.write(t_corpus)
        #         print(t_corpus)
