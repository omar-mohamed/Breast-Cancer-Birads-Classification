from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from generator import AugmentedImageSequence
from configs import argHandler  # Import the default arguments
from model_utils import set_gpu_usage, get_evaluation_metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics


FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

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
        shuffle_on_epoch_end=False,
    )


train_generator = get_generator(FLAGS.train_csv)
test_generator = get_generator(FLAGS.test_csv)

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)

def get_metrics_from_generator(generator,threshold=0.5, verbose=1):
    y_hat = visual_model.predict_generator(generator, steps=generator.steps, workers=FLAGS.generator_workers,
                                           max_queue_size=FLAGS.generator_queue_length, verbose=verbose)
    y = generator.get_y_true()
    get_evaluation_metrics(y_hat, y, FLAGS.classes,threshold=threshold)

if FLAGS.multi_label_classification:
    visual_model.compile(loss='binary_crossentropy',
                         metrics=[metrics.BinaryAccuracy(threshold=FLAGS.multilabel_threshold)])

    print("***************Train Metrics*********************")
    get_metrics_from_generator(train_generator, FLAGS.multilabel_threshold)
    print("***************Test Metrics**********************")
    get_metrics_from_generator(test_generator, FLAGS.multilabel_threshold)

else:
    visual_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    visual_model.evaluate_generator(
        generator=test_generator,
        steps=test_generator.steps,
        workers=FLAGS.generator_workers,
        max_queue_size=FLAGS.generator_queue_length,
        verbose=1)
