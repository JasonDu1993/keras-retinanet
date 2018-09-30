import os
import six
import csv
import numpy as np
from keras.callbacks import Callback
from collections import deque
from collections import OrderedDict
from collections import Iterable


class SaveRetinanetLossValue(Callback):
    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self.writer = None
        self.csv_file = None
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''

    def on_train_begin(self, logs=None):
        if self.append:
            self.csv_file = open(self.filename, 'a' + self.file_flags)
            self.writer = csv.writer(self.csv_file)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(
                ["epoch", "loss", "val_loss", "regression_loss", "classification_loss", "val_regression_loss",
                 "val_classification_loss"])
            self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs["loss"]
        val_loss = logs['val_loss']
        regression_loss = logs["regression_loss"]
        val_regression_loss = logs["val_regression_loss"]
        classification_loss = logs["classification_loss"]
        val_classification_loss = logs["val_classification_loss"]
        # print("loss", epoch, loss, val_loss)
        # print("regression_loss", regression_loss)
        self.writer.writerow(
            [epoch, loss, val_loss, regression_loss, classification_loss, val_regression_loss, val_classification_loss])
        self.csv_file.flush()
        # def handle_value(k):
        #     is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        #     if isinstance(k, six.string_types):
        #         return k
        #     elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
        #         return '"[%s]"' % (', '.join(map(str, k)))
        #     else:
        #         return k
        #
        # if self.keys is None:
        #     self.keys = sorted(logs.keys())
        #
        # if self.model.stop_training:
        #     # We set NA so that csv parsers do not fail for this last epoch.
        #     logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        #
        # if not self.writer:
        #     class CustomDialect(csv.excel):
        #         delimiter = self.sep
        #
        #     self.writer = csv.DictWriter(self.csv_file,
        #                                  fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
        #     if self.append_header:
        #         self.writer.writeheader()
        #
        # row_dict = OrderedDict({'epoch': epoch})
        # row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        # self.writer.writerow(row_dict)
        # self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class CustomModelCheckpoint(Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
        path1 = self.path
        path1 = path1.format(epoch=epoch, loss=loss, val_loss=val_loss)
        self.model.save_weights(path1, overwrite=False)

        self.best_loss = val_loss
