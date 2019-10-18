from __future__ import absolute_import, division

import importlib
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from dense_classifier import get_classifier

class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            DenseNet169=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            DenseNet201=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),            
            Xception=dict(
                input_shape=(299, 299, 3),
                module_name="xception",
                last_conv_layer="block14_sepconv2_bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
            MobileNet=dict(
                input_shape=(224, 224, 3),
                module_name="mobilenet",
                last_conv_layer="conv_pw_13_bn",
            ),
            MobileNetV2=dict(
                input_shape=(224, 224, 3),
                module_name="mobilenet_v2",
                last_conv_layer="Conv_1_bn",
            ),            
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]
    
    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, FLAGS):

        if FLAGS.use_imagenet_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                "tensorflow.keras.applications.{}".format(self.models_[FLAGS.visual_model_name]['module_name'])
            ),
            FLAGS.visual_model_name)
        
        input_shape = FLAGS.image_target_size
        if input_shape is None:
            input_shape = self.models_[FLAGS.visual_model_name]["input_shape"]
        
        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling=FLAGS.final_layer_pooling)

        if FLAGS.use_chexnet_weights and FLAGS.visual_model_name == 'DenseNet121':
            predictions = Dense(14, activation="sigmoid", name="predictions")(base_model.output)
            base_model = Model(inputs=img_input, outputs=predictions)
            base_model.load_weights(FLAGS.chexnet_weights_path)
            print(f"loaded chexnet weights: {FLAGS.chexnet_weights_path}")


        for i in range(FLAGS.pop_conv_layers):
             base_model._layers.pop()

        if FLAGS.conv_layers_to_train != -1:
            for i in range(len(base_model.layers)-FLAGS.conv_layers_to_train):
                base_model.layers[i].trainable = False  
                
        base_model.outputs=base_model.layers[-1].output
        base_model_output =  base_model.outputs
        
        loaded_model = base_model
        classifier = None
        if FLAGS.classes is not None and (not FLAGS.use_chexnet_weights or FLAGS.visual_model_name != 'DenseNet121' or FLAGS.pop_conv_layers>0):
            output_unrolled_length = 1
            for dimension in base_model_output.shape[1:]:
                output_unrolled_length *= int(dimension)            
                
            classifier = get_classifier(output_unrolled_length,FLAGS.multi_label_classification, FLAGS.classifier_layer_sizes,len(FLAGS.classes))
            predictions = classifier(base_model_output)
            loaded_model = Model(inputs=img_input, outputs=predictions)
        
        loaded_model.summary()
        if classifier is not None:
            classifier.summary()
        
        return loaded_model
