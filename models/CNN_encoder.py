import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, units,num_classes=5):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x
