import json
useGPU = json.loads(open('config.json',).read())['useGPU']

import tensorflow
if not eval(useGPU):
    print('Not using GPU.')
    tensorflow.config.set_visible_devices([], 'GPU')
print('Tensorflow version:', tensorflow.__version__)

from tensorflow import keras
from tensorflow.keras import layers


# Testing:
visionModel = keras.applications.ResNet50()
print(visionModel)