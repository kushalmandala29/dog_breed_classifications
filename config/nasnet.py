from keras.applications.nasnet import NASNetLarge, preprocess_input

class nasnet_p:
    def __init__(self):
        self.nasnet_preprocessor = preprocess_input