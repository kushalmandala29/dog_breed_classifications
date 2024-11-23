from keras.applications.inception_v3 import InceptionV3, preprocess_input

class inception_p:
    def __init__(self):
        self.inception_preprocessor = preprocess_input