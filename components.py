import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import Sequential
from keras.callbacks import EarlyStopping
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from keras.utils import to_categorical
# from keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing.image import load_img
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
import pandas as pd
from config.xception import xception_p
# import os
from config.inception import inception_p
from config.inc_resnet import inc_resnet_p
from config.nasnet import nasnet_p

class Models:
    def __init__(self):
        
        self.model=load_model('model.h5',compile=False)
        self.inception_preprocessor = inception_p().inception_preprocessor
        self.xception_preprocessor = xception_p().xception_preprocessor
        self.inc_resnet_preprocessor = inc_resnet_p().inception_preprocessor
        self.nasnet_preprocessor = nasnet_p().nasnet_preprocessor
        self.img_size = (331, 331, 3)
        self.labels = pd.read_csv('data/labels.csv')
        self.classes = sorted(list(set(self.labels['breed'])))
        

    
    def get_features(self,model_name, model_preprocessor, input_size, data):

        input_layer = Input(input_size)
        preprocessor = Lambda(model_preprocessor)(input_layer)
        base_model = model_name(weights='imagenet', include_top=False,
                                input_shape=input_size)(preprocessor)
        avg = GlobalAveragePooling2D()(base_model)
        feature_extractor = Model(inputs = input_layer, outputs = avg)
        
        #Extract feature.
        feature_maps = feature_extractor.predict(data, verbose=1)
        print('Feature maps shape: ', feature_maps.shape)
        return feature_maps

    def extact_features(self,data):
        inception_features = self.get_features(InceptionV3, self.inception_preprocessor, self.img_size, data)
        xception_features = self.get_features(Xception, self.xception_preprocessor, self.img_size, data)
        nasnet_features = self.get_features(NASNetLarge, self.nasnet_preprocessor, self.img_size, data)
        inc_resnet_features = self.get_features(InceptionResNetV2, self.inc_resnet_preprocessor, self.img_size, data)

        final_features = np.concatenate([inception_features,
                                        xception_features,
                                        nasnet_features,
                                        inc_resnet_features],axis=-1)
        
        # print('Final feature maps shape', final_features.shape)
        
        #deleting to free up ram memory
        del inception_features
        del xception_features
        del nasnet_features
        del inc_resnet_features

        
        
        return final_features
    

    def predict(self,Image):
        # img_g = load_img(Image,target_size = self.img_size)
        img_g = np.expand_dims(Image, axis=0)
        test_features = self.extact_features(img_g)
        predg = self.model.predict(test_features)

        return self.classes[np.argmax(predg[0])], np.max(predg[0]) * 100
        