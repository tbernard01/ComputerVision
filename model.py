import os 
import numpy as np 
import tensorflow as tf 
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class CTCModel(): 
    def __init__(self,
                 batch_size,  
                 input_size, 
                 vocab_size, 
                 nb_epoch = 5, 
                 architecture="model"):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.model = None
        self.nb_epoch = nb_epoch
        self.architecture = globals()[architecture]


    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        """ Load a model with checkpoint file"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)


    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """Setup the list of callbacks for the model"""

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=20,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks
    
    def compile(self, learning_rate=None, initial_step=0):
        """
        Configures the HTR Model for training/predict.
        :param optimizer: optimizer for training
        """

        # define inputs, outputs and optimizer of the chosen architecture
        inputs, outputs = self.architecture(self.batch_size, self.input_size, self.vocab_size + 1)

        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.vocab_size + 1, initial_step=initial_step)
            self.learning_schedule = True
        else:
            self.learning_schedule = False

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        # create and compile
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss_lambda_func)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=2,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):


        if callbacks and self.learning_schedule:
            callbacks = [x for x in callbacks if not isinstance(x, ReduceLROnPlateau)]

        out = self.model.fit(x=x, y=y, epochs=epochs,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=50,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):
        """
        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """

        if verbose == 1:
            print("Model Predict")

        out = self.model.predict(x=x, batch_size=batch_size, verbose=verbose)

        if not ctc_decode:
            return np.log(out.clip(min=1e-8)), []

        steps_done = 0
        steps = out.shape[0]
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []

        while steps_done < steps:
            until = steps_done + batch_size

            x_test = out [steps_done:until] 
            x_test_len = [input_length for _ in range(len(x_test))]

            decode, log = K.ctc_decode(x_test,
                                       x_test_len)
 

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

        return (predicts, probabilities)


    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):
        """Function for computing the CTC loss"""

        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with -1
        # so sum of non-zero gives number of characters in this string
        label_length = tf.expand_dims(tf.math.reduce_sum(tf.where(y_true==-1, 0, 1), axis=-1), -1)
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)
        return loss

 
        

def model(batch_size, 
          max_len, 
          nb_labels,
          kernel_sizes=[3, 4, 5], 
          filters=200, 
          rnn_cells=100):
        
        input_layer = layers.Input(shape=(None, max_len))
        convs = []
        
        for kernel_size in kernel_sizes:
            convs.append(
                tf.keras.layers.Conv1D(
                    filters,
                    kernel_size=kernel_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                    bias_initializer=tf.constant_initializer(value=0.1),
                    activation=None,
                    padding="valid",
                    name="conv1d_{}".format(kernel_size),
                )
            )
        pooling = layers.GlobalAveragePooling1D()
        lstm = layers.LSTM(rnn_cells, return_sequences=True)
        dense_lstm = layers.Dense(nb_labels)
    

        convs = tf.concat([conv(input_layer) for conv in convs], 1)
        print(convs)
        rnn_output = lstm(convs)
        rnn_output = dense_lstm(rnn_output)
        output = tf.nn.softmax(rnn_output)
        return input_layer, output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom schedule of the learning rate with warmup_steps.
    From original paper "Attention is all you need".
    """

    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype="float32")
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)