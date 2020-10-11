#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 16:40:05 2020

@author: b_suharta
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class player:
    def __init__(self, filename):
        self.filename = filename


    def predictionplayer(self):
        model = load_model("lenet-model.h5")

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = "sachin"
            print(prediction)
            return [{"image": prediction}]

        elif result[0][1] == 1:
            prediction = "sourav"
            return [{"image": prediction}]

        elif result[0][2] == 1:
            prediction = "virat"
            return [{"image": prediction}]

        else:
            prediction = "yuvraj"
            return [{"image": prediction}]
