from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from generator import AugmentedImageSequence
from configs import argHandler  # Import the default arguments
from model_utils import load_model, get_accuracy_from_generator, set_gpu_usage

FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

test_generator = AugmentedImageSequence(
    dataset_csv_file=FLAGS.test_csv,
    label_columns=FLAGS.csv_label_columns,
    class_names=FLAGS.classes,
    multi_label_classification=FLAGS.multi_label_classification,
    source_image_dir=FLAGS.image_directory,
    batch_size=FLAGS.batch_size,
    target_size=FLAGS.image_target_size,
    shuffle_on_epoch_end=True,
)
# load classifier from saved weights or get a new one
if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path, FLAGS.load_model_name)
else:
    visual_model = model_factory.get_model(FLAGS)

test_accuracy = get_accuracy_from_generator(visual_model, test_generator, FLAGS.multi_label_classification)
print('Test Accuracy: %.2f' % (test_accuracy))
