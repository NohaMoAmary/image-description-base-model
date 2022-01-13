# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
from pickle import load
import tensorflow
import os
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

class Predictor(cog.Predictor):
    def setup(self):
      """Load the model into memory to make running multiple predictions efficient"""
      # load the tokenizer
      self.tokenizer = load(open('./tokenizer.pkl', 'rb'))
      # load the model
      self.model = load_model('./model_18.h5')
    
    
    @cog.input("input", type=cog.Path, help="Image to descripe")
    def predict(self, input):
        """Run a single prediction on the model"""
        # load the photo
        image = load_img(input, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # load and prepare the photograph
        # extract features from each photo in the directory
        model = VGG16()
        # re-structure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        feature = model.predict(image, verbose=0)
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
        
        
