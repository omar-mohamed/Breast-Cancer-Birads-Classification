import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from chexnet_wrapper import ChexnetWrapper
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from utility import get_sample_counts
from tensorflow.python.eager.context import graph_mode
import numpy as np

config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)

class_names = cp["Classification_Model"].get("class_names").split(",")
num_classes = cp["Classification_Model"].getint("num_classes")

image_source_dir = cp["Data"].get("image_source_dir")
data_dir = cp["Data"].get("data_dir")
all_data_csv = cp['Data'].get('all_data_csv')
testing_csv = cp['Data'].get('training_set_csv')

image_dimension = cp["Chexnet_Default"].getint("image_dimension")

batch_size = cp["Classification_Model_Inference"].getint("batch_size")
testing_counts = get_sample_counts(data_dir, testing_csv)


units = cp["Classification_Model"].getint("units")

checkpoint_path = cp["Classification_Model_Train"].get("ckpt_path")

# compute steps
steps = int(testing_counts / batch_size)

print(f"** test: {steps} **")

print("** load test generator **")

data_generator = AugmentedImageSequence(
    dataset_csv_file=os.path.join(data_dir, testing_csv),
    class_names=class_names,
    source_image_dir=image_source_dir,
    batch_size=batch_size,
    target_size=(image_dimension, image_dimension),
    steps=steps,
    shuffle_on_epoch_end=False,
)


encoder = CNN_Encoder(units)
optimizer = tf.keras.optimizers.Adam()

with graph_mode():
    chexnet = ChexnetWrapper()

ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))


def evaluate(image_tensor):
    image_tensor=tf.reshape(image_tensor,[image_tensor.shape[0],-1])

    logits = encoder(image_tensor)

    predictions = tf.nn.softmax(logits)

    return np.argmax(predictions)




total_loss = 0

targets=[]
predictions=[]
for batch in range(data_generator.steps):
    print("Batch: {}".format(batch))
    img, target, img_path = data_generator.__getitem__(batch)
    with graph_mode():
        img_tensor = chexnet.get_visual_features(img)
    prediction = evaluate(img_tensor)
    targets.append(target)
    predictions.append(prediction)


print("accuracy: {0:0.2f}%".format(np.sum(np.array(predictions)==np.array(targets))/len(targets)))


