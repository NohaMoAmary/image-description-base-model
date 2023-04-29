from typing import Any

from pickle import load
import tensorflow as tf
import os
import numpy as np
from numpy import argmax
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    DirectoryIterator,
    array_to_img,
    save_img,
)
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cog import BasePredictor, Input, Path



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # load the tokenizer
        self.tokenizer = load(open('./tokenizer.pkl', 'rb'))
        # load the model
        self.model = load_model('./model_18.h5')
    
    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to Descripe")) -> Any:
        
        # load and prepare the photograph
        # extract features from each photo in the directory
        #modelv = VGG16()
        modelv = VGG16(weights="vgg16_weights_tf_dim_ordering_tf_kernels.h5")
        # re-structure the model
        modelv = Model(inputs=modelv.inputs, outputs=modelv.layers[-2].output)

        
        """Run a single prediction on the model"""
        # Preprocess the image
        img = keras_image.load_img(image, target_size=(224, 224))
        # convert the image pixels to a numpy array
        x = keras_image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        # reshape data for the model
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        # prepare the image for the VGG model
        x = preprocess_input(x)
        feature = modelv.predict(x, verbose=0)


        # generate description
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(34):
          # integer encode input sequence
          sequence = self.tokenizer.texts_to_sequences([in_text])[0]
          # pad input
          sequence = pad_sequences([sequence], maxlen=34)
          # predict next word
          yhat = self.model.predict([feature,sequence], verbose=0)
          # convert probability to integer
          yhat = argmax(yhat)
          # map integer to word
          word = ""
          for word1, index in self.tokenizer.word_index.items():
            if index == yhat:
              word =word1
          # stop if we cannot map the word
          if word is None:
            break
          # append as input for generating the next word
          in_text += ' ' + word
          # stop if we predict the end of the sequence
          if word == 'endseq':
            break

        #Remove startseq and endseq
        query = in_text
        stopwords = ['startseq','endseq']
        querywords = query.split()
        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        result = ' '.join(resultwords)
        # Return the result 
        return result 

