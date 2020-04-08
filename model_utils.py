from __future__ import absolute_import, division

from skimage.transform import resize
from tensorflow.keras.models import model_from_json
import os
import numpy as np
from tensorflow.keras import backend as K
import importlib
import efficientnet.tfkeras
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, hamming_loss, confusion_matrix, accuracy_score, classification_report
from generator import AugmentedImageSequence
import math

def set_gpu_usage(gpu_memory_fraction):
    pass
    # if gpu_memory_fraction <= 1 and gpu_memory_fraction > 0:
    #     config = tf.ConfigProto(allow_soft_placement=True)
    #     config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    #     sess = tf.Session(config=config)
    # elif gpu_memory_fraction == 0:
    #     sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    # K.set_session(sess)

def get_generator(csv_path,FLAGS, data_augmenter=None):
    return AugmentedImageSequence(
        dataset_csv_file=csv_path,
        label_columns=FLAGS.csv_label_columns,
        class_names=FLAGS.classes,
        multi_label_classification=FLAGS.multi_label_classification,
        source_image_dir=FLAGS.image_directory,
        batch_size=FLAGS.batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=data_augmenter,
        shuffle_on_epoch_end=False,
    )
def get_optimizer(optimizer_type, learning_rate, lr_decay=0):
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
def get_accuracy_from_generator(model, generator, multi_label_classification,threshold):
    true_predictions_count = 0.0
    data_count = 0.0
    # max=0
    for step in range(generator.steps):
        (batch_x, batch_y) = next(generator)
        predictions = model.predict(batch_x)
        if multi_label_classification:
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0
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



def get_multilabel_evaluation_metrics(pred, labels, class_names, threshold=0.5):
    current_auroc = []
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(labels[:, i], pred[:, i])
        except ValueError:
            score = 0
        current_auroc.append(score)
        print(f"{i + 1}. {class_names[i]}: {score}")
    print("*********************************")

    mean_auroc = np.mean(current_auroc)
    print(f"mean auroc: {mean_auroc}")

    prec, rec, fscore, support = precision_recall_fscore_support(labels, pred >= threshold, average='macro')
    AP = average_precision_score(labels, pred)
    exact_accuracy = accuracy_score(labels, pred >= threshold)
    ham_loss = hamming_loss(labels, pred >= threshold)
    print(f"precision:{prec:.2f}, recall: {rec:.2f}, fscore: {fscore:.2f}, AP: {AP:.2f}, exact match accuracy: {exact_accuracy:.2f}, hamming loss: {ham_loss:.2f}")
    return mean_auroc, prec,rec,fscore, AP , exact_accuracy, ham_loss

def get_sample_counts(labels):

    total_count = labels.shape[0]
    positive_counts = np.sum(labels, axis=0)
    classes=[]
    for i in range(labels.shape[1]):
        classes.append(str(i))
    class_positive_counts = dict(zip(classes, positive_counts))
    return total_count, class_positive_counts

# predict on data from generator and calculate accuracy
def get_evaluation_metrics(predictions, labels, class_names):
    print(classification_report(labels, predictions, target_names=class_names))
    print("*******Confusion matrix*********")
    print(confusion_matrix(labels, predictions))
    print("\nAccuracy: %.2f" % accuracy_score(labels, predictions))



def get_multilabel_class_weights(labels, multiply):

    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    total_counts, class_positive_counts = get_sample_counts(labels)
    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_class_weights(labels_count,mu=0.15):
    total = np.sum(labels_count)
    class_weight = dict()

    for key in range(len(labels_count)):
        score = math.log(mu*total/float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight