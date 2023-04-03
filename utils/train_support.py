import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

class DataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, augmentations=None):
        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        self.x = np.array(x_set)
        self.y= np.array(y_set)
        self.batch_size= batch_size
        self.augment = augmentations # seems to be albumentation.Compose instance
        self._total_called = 0
        self.indices = list(np.arange(self.x.__len__()))
    def __len__(self):
        try:
            return int(np.ceil(len(self.x) / float(self.batch_size)))
        except TypeError:
            return int(np.ceil(self.x.__len__() / float(self.batch_size)))
    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        if self.augment == None:
            return np.array([img_to_array(load_img(file_name, target_size=(64, 64)), dtype=np.float32) / 255 for file_name in batch_x]),\
                   np.array(batch_y)
        else:
            return np.array([self.augment(image=img_to_array(load_img(file_name, target_size=(64, 64)), dtype=np.float32) / 255)["image"]
                         for file_name in batch_x]),\
                   np.array(batch_y)
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def visualize_training_history(history, save_path):
    # list all data in history
    print(history.history.keys())
    keys = history.history.keys()
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    # summarize history for accuracy
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(save_path,'model_accuracy.png'))
    # summarize history for loss
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(save_path,'model_loss.png'))