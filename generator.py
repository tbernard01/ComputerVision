import utils
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator():
    def __init__(self, path='./data/', batch_size=50):
        
        self.batch_size=batch_size

        train, test, valid = {}, {}, {}
        self.size = {'train':0, 'test':0, 'valid':0 }
        self.steps = {'train':0, 'test':0, 'valid':0 }
        self.index = {'train':0, 'test':0, 'valid':0 }
        
        train['hw_string'], train['labels'], self.all_labels = utils.read_data(path, set_to_read='train')
        valid['hw_string'], valid['labels'], _ = utils.read_data(path, set_to_read='valid')
        # test['hw_string'], test['labels'], _ = utils.read_data(path, set_to_read='test')

        self.dataset = {'train': train, 'test':test, 'valid':valid}
        
        self.max_len = max([ d.shape[0] for d in train['hw_string'] ])
        self.nb_labels = len(self.all_labels)
        
        for pt in ['train', 'valid']:
            self.size[pt] = len(self.dataset[pt]['hw_string'])
            self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))

    def pipeline_preparation(self, hw_string, labels, all_labels):

        string_lengths = tf.cast([len(string) for string in hw_string], tf.int32)
        hw_string = pad_sequences(hw_string, self.max_len, value=0)
        
        labels = [[all_labels[l] for l in label] for label in  labels]
        labels = pad_sequences(labels, max([len(d) for d in labels]), value=-1) + 1
        label_lengths = tf.cast([len(lab) for lab in labels], tf.int32)

        return tf.transpose(hw_string, [0, 2, 1]), string_lengths, labels, label_lengths

    def next_train_batch(self):

        self.index['train'] = 0

        x, x_length, y,  y_length = self.pipeline_preparation(self.dataset['train']['hw_string'], 
                                                         self.dataset['train']['labels'], 
                                                         self.all_labels
                                                         )

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = index + self.batch_size

            x_train  = x[index:until]
            x_train_length = x_length[index:until]
            y_train = y[index:until]
            y_train_length = y_length[index:until] 
            

            self.index['train'] = until


            yield (x_train, y_train)

    def next_valid_batch(self):

        self.index['valid'] = 0

        x, x_length, y,  y_length = self.pipeline_preparation(self.dataset['valid']['hw_string'], 
                                                         self.dataset['valid']['labels'], 
                                                         self.all_labels
                                                         )

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = index + self.batch_size

            x_train  = x[index:until]
            x_train_length = x_length[index:until]
            y_train = y[index:until]
            y_train_length = y_length[index:until] 
            

            self.index['valid'] = until


            yield (x_train, y_train)

