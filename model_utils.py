from __future__ import absolute_import, division

from skimage.transform import resize
from tensorflow.keras.models import model_from_json
import os
import numpy as np
import importlib
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix


def set_gpu_usage(gpu_memory_fraction):
    pass
    # if gpu_memory_fraction <= 1 and gpu_memory_fraction > 0:
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    #     sess = tf.Session(config=config)
    # elif gpu_memory_fraction == 0:
    #     sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    # K.set_session(sess)


def get_optimizer(optimizer_type, learning_rate, lr_decay):
    optimizer_class = getattr(importlib.import_module("tensorflow.keras.optimizers"), optimizer_type)
    optimizer = optimizer_class(lr=learning_rate, decay=lr_decay)
    return optimizer


def save_model(model, save_path, model_name):
    try:
        os.makedirs(save_path)
    except:
        print("path already exists")

    path = os.path.join(save_path, model_name)
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}.json".format(path), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}.h5".format(path))
    print("Saved model to disk")


def load_model(load_path, model_name):
    path = os.path.join(load_path, model_name)

    # load json and create model
    json_file = open('{}.json'.format(path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    loaded_model.load_weights("{}.h5".format(path))
    print("Loaded model from disk")
    return loaded_model


def classify_image(img, model, multi_label_classification, target_size=(224, 224, 3)):
    # resize
    img = img / 255.
    img = resize(img, target_size)
    batch_x = np.expand_dims(img, axis=0)
    # normalize
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    batch_x = (batch_x - imagenet_mean) / imagenet_std
    # predict
    predictions = model.predict(batch_x)
    if multi_label_classification:
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
    else:
        predictions = np.argmax(predictions, axis=1)
    return predictions


# predict on data from generator and calculate accuracy
def get_accuracy_from_generator(model, generator, multi_label_classification):
    true_predictions_count = 0.0
    data_count = 0.0
    for step in range(generator.steps):
        (batch_x, batch_y, _) = next(generator)
        predictions = model.predict(batch_x)
        if multi_label_classification:
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            true_predictions_count += np.sum((predictions == batch_y).all(axis=1))
        else:
            predictions = np.argmax(predictions, axis=1)
            true_predictions_count += np.sum(predictions == batch_y)
        data_count += batch_x.shape[0]
    accuracy = (true_predictions_count / data_count) * 100.0
    return accuracy


def get_accuracy(predictions, labels, multi_label_classification):
    if multi_label_classification:
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        true_predictions_count = np.sum((predictions == labels).all(axis=1))
    else:
        predictions = np.argmax(predictions, axis=1)
        true_predictions_count = np.sum(predictions == labels)
    return (true_predictions_count / labels.shape[0]) * 100.0
