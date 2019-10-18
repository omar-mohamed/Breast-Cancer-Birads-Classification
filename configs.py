class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}

    def setDefaults(self):
        self.define('train_csv', './data/training_set.csv',
                    'path to training csv containing the images names and the labels')
        self.define('test_csv', './data/testing_set.csv',
                    'path to testing csv containing the images names and the labels')
        self.define('image_directory', './data/images',
                    'path to folder containing the patient folders which containg the images')
        self.define('visual_model_name', 'MobileNetV2',
                    'select from (VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, Xception, ResNet50, InceptionV3, InceptionResNetV2, NASNetMobile, NASNetLarge, MobileNet, MobileNetV2)')
        self.define('use_chexnet_weights', True,
                    'use pre-trained chexnet weights. Note only works with DenseNet121. If you use this option without popping layers it will have the classifier intact')
        self.define('chexnet_weights_path', 'pretrained_models/chexnet_densenet121_weights.h5', 'chexnet weights path')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('csv_label_columns', ['BIRADS'], 'the name of the label columns in the csv')
        self.define('classes', ['BIRAD-1', 'BIRAD-2', 'BIRAD-3', 'BIRAD-4', 'BIRAD-5'],
                    'the names of the output classes')

        self.define('multi_label_classification', True,
                    'determines if this is a multi classification problem or not. It affects the loss function')

        self.define('classifier_layer_sizes', [],
                    'a list describing the hidden layers of the classifier. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('conv_layers_to_train', 0,
                    'the number of layers that should be trained in the visual model counting from the end. -1 means train all and 0 means freezing the visual model')
        self.define('use_imagenet_weights', True, 'initialize the visual model with pretrained weights on imagenet')
        self.define('pop_conv_layers', 0,
                    'number of layers to be popped from the visual model. Note that the imagenet classifier is removed by default so you should not take them into considaration')
        self.define('final_layer_pooling', 'avg', 'the pooling to be used as a final layer to the visual model')
        self.define('load_model_path', '',
                    'a path containing the checkpoints. If provided with load_model_name the system will continue the training from that point or use it in testing.')
        self.define('load_model_name', '',
                    'the name of the chekpoint file. If provided with load_model_name the system will continue the training from that point or use it in testing.')
        self.define('save_model_path', './saved_model',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('learning_rate', 1e-5, 'The optimizer learning rate')
        self.define('learning_rate_decay', 0, 'Learning rate decay over each update')
        self.define('gpu_percentage', 0.95, 'gpu utilization. If 0 it will use the cpu')
        self.define('batch_size', 4, 'batch size for training and testing')

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        print('')
        exit()
