import os
import numpy as np
import tensorflow as tf
from keras import backend as K
LOG = 1
# This custom callback is used to extract get recall per color category every epoch
class LearnedAccuracyWriter(tf.keras.callbacks.Callback):

    def __init__(self, colours, training_set, iteration, patience, results_dir):
        self.colours = colours
        self.output_dir = results_dir
        self.num_classes = len(colours)
        self.training_set = training_set
        self.accuracy = 0
        self.iteration = iteration
        self.patience = patience
        self.unincreased_epochs = 0
        self.accuracy_tracker = {}
        self.recall_threshold = 0.85


    def on_epoch_end(self, epoch, logs=None):
        print('lr', K.eval(self.model.optimizer.lr))
        epoch = epoch + 1
        # Convert one-hot to index

        next_set_x, next_set_y = self.training_set.next()
        y_pred = self.model.predict(next_set_x)
        y1 = np.argmax(y_pred, axis=1)
        y2 = np.argmax(next_set_y, axis=1)
        correct = [0] * self.num_classes
        incorrect = [0] * self.num_classes

        for i in range(len(y2)):
            if y1[i] == y2[i]:
                correct[int(y2[i])] += 1
            else:
                incorrect[int(y2[i])] += 1

        for color_index in range(self.num_classes):
            if (correct[color_index] + incorrect[color_index]) != 0 and correct[color_index] / (correct[color_index] + incorrect[color_index]) >= self.recall_threshold:
                if self.colours[color_index] not in self.accuracy_tracker or self.accuracy_tracker[self.colours[color_index]] == None:
                    self.accuracy_tracker[self.colours[color_index]] = epoch
            else:
                self.accuracy_tracker[self.colours[color_index]] = None
            if LOG:
                print(
                    "{}={}/{}".format(self.colours[color_index], correct[color_index], correct[color_index] + incorrect[color_index]))
        if LOG:
            print('{}'.format(self.accuracy_tracker))

        if logs['val_accuracy'] > self.accuracy:
            self.unincreased_epochs = 0
            self.accuracy = logs['val_accuracy']
        else:
            self.unincreased_epochs = self.unincreased_epochs + 1
        if self.unincreased_epochs == self.patience or (len(self.accuracy_tracker) > 0 and None not in list(self.accuracy_tracker.values())):
            self.model.stop_training = True
            self.write_results(epoch)

    def write_results(self, epoch):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        with open('{}/{}.txt'.format(self.output_dir, self.iteration), 'w') as fh:
            fh.write('{}\n'.format(self.accuracy_tracker))
            fh.write('{}\n'.format(epoch))
