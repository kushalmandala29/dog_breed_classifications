from components import Models
from tensorflow.keras.models import load_model
if __name__ == '__main__':
    
    # model=load_model('model.h5',compile=False)
    # print(model)
    bread,accuracy=Models().predict('data/test/ffd87c3e44faa0e3ee5fbbdc4c63b59b.jpg')
    print(bread,accuracy)
    # import tensorflow as tf
    # print(tf.config.list_physical_devices('GPU'))
