import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from chexnet_wrapper import ChexnetWrapper
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from utility import get_sample_counts
import time
from tensorflow.python.eager.context import eager_mode, graph_mode
from augmenter import augmenter
import matplotlib.pyplot as plt

config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["Classification_Model"].get("class_names").split(",")
num_classes = cp["Classification_Model"].getint("num_classes")

image_source_dir = cp["Data"].get("image_source_dir")
data_dir = cp["Data"].get("data_dir")
training_csv=cp['Data'].get('training_set_csv')

image_dimension = cp["Chexnet_Default"].getint("image_dimension")

batch_size = cp["Classification_Model_Train"].getint("batch_size")
training_counts = get_sample_counts(data_dir, training_csv)
EPOCHS = cp["Classification_Model_Train"].getint("epochs")

units = cp["Classification_Model"].getint("units")

checkpoint_path = cp["Classification_Model_Train"].get("ckpt_path")
continue_from_last_ckpt=cp["Classification_Model_Train"].getboolean("continue_from_last_ckpt")
# compute steps
steps = int(training_counts / batch_size)
print(f"** train_steps: {steps} **")

print("** load training generator **")


data_generator = AugmentedImageSequence(
    dataset_csv_file=os.path.join(data_dir, training_csv),
    class_names=class_names,
    source_image_dir=image_source_dir,
    batch_size=batch_size,
    target_size=(image_dimension, image_dimension),
    augmenter=augmenter,
    steps=steps,
    shuffle_on_epoch_end=True,
)


encoder = CNN_Encoder(units,num_classes)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)


loss_plot = []


@tf.function
def train_step(img_tensor, labels):
    loss = 0
    img_tensor=tf.reshape(img_tensor,[img_tensor.shape[0],-1])
    with tf.GradientTape() as tape:
        logits = encoder(img_tensor)
        predictions= tf.nn.softmax(logits)
        loss += loss_function(labels, predictions)

    trainable_variables = encoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


with graph_mode():
    chexnet = ChexnetWrapper()


ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint and continue_from_last_ckpt:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for batch in range(data_generator.steps):
        img, target, paths = data_generator.__getitem__(batch)
        with graph_mode():
            img_tensor = chexnet.get_visual_features(img)

        print("batch: {}".format(batch))

        batch_loss = train_step(img_tensor, target)
        total_loss += batch_loss

        if batch % 5 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy()))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / data_generator.steps)

    if epoch % 1 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss / data_generator.steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig("loss.png")

# plt.show()
