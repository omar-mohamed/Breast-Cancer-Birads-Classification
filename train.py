from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from generator import AugmentedImageSequence
from configs import argHandler  # Import the default arguments
from model_utils import save_model, load_model, get_accuracy, get_accuracy_from_generator, get_optimizer
import numpy as np
from augmenter import augmenter

FLAGS = argHandler()
FLAGS.setDefaults()

model_factory = ModelFactory()


# load training and test set file names

def get_generator(csv_path, data_augmenter=None):
    return AugmentedImageSequence(
        dataset_csv_file=csv_path,
        label_columns=FLAGS.csv_label_columns,
        class_names=FLAGS.classes,
        multi_label_classification=FLAGS.multi_label_classification,
        source_image_dir=FLAGS.image_directory,
        batch_size=FLAGS.batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=data_augmenter,
        shuffle_on_epoch_end=True,
    )


train_generator = get_generator(FLAGS.train_csv, augmenter)

test_generator = get_generator(FLAGS.test_csv)

# load classifier from saved weights or get a new one
if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path, FLAGS.load_model_name)
    epoch_number = int(FLAGS.load_model_name.split('_')[-1])
else:
    visual_model = model_factory.get_model(FLAGS)
    epoch_number = 0

opt = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate, FLAGS.learning_rate_decay)

if FLAGS.multi_label_classification:
    visual_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
else:
    visual_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

accuracies = []
steps = 0
best_test_accuracy = 0
best_test_accuracy_epoch = 0

accuracies = []
losses = []
# run training
while True:

    (batch_x, batch_y, _) = next(train_generator)
    batch_loss = visual_model.train_on_batch(batch_x, batch_y)
    batch_predictions = visual_model.predict(batch_x)
    accuracy = get_accuracy(batch_predictions, batch_y, FLAGS.multi_label_classification)
    accuracies.append(accuracy)
    losses.append(np.mean(batch_loss))
    if steps % 5 == 0:
        print('Step: %d' % steps)
        print('Batch Accuracy: %.2f' % accuracy)
        print('Batch Loss: %.2f' % np.mean(batch_loss))

    steps += 1
    if (steps % train_generator.steps == 0):
        epoch_number += 1
        test_accuracy = get_accuracy_from_generator(visual_model, test_generator, FLAGS.multi_label_classification)
        print('Epoch: %d' % epoch_number)
        print('Training Accuracy: %.2f' % (np.mean(np.array(accuracies))))
        print('Training Loss: %.2f' % (np.mean(np.array(losses))))
        print('Test Accuracy: %.2f' % test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_accuracy_epoch = epoch_number
        accuracies = []
        losses = []
        save_model(visual_model, FLAGS.save_model_path, "model_epoch_{}".format(str(epoch_number)))
        print('Best Test Accuracy: %.2f in epoch: %d' % (best_test_accuracy, best_test_accuracy_epoch))
        if epoch_number == FLAGS.num_epochs:
            break
