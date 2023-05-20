from flask import Flask, jsonify
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import keras.utils as image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


FOLDER = 'uploadedfiles'

# Model saved with Keras model.save()
MODEL_PATH = './mobileNet_5classes9829(3)/mobileNet_5classes9829(3)'
# MODEL_PATH = 'mobileNet_5classes91(2)'

# Load your trained model
model = load_model(MODEL_PATH)

model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save('')
def prediction(img_path, model):

    from tensorflow.keras.preprocessing import image
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    # Setting Training Hyperparameters
    img_width =img_height =224

    # create a sharpening kernel
    sharpen_filter =np.array([[0 ,-4 ,0],
                             [-4 ,17 ,-4],
                             [0 ,-4 ,0]])

    org_img = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range

    # applying kernels to the BGR image to get the sharpened image
    sharp_image =cv2.filter2D(img_tensor ,-1 ,sharpen_filter)

    sharp_image =sharp_image.reshape(1 ,img_width ,img_height ,3)
    sharp_image /= 255.  # Normalize to [0,1] for plt.imshow application
    # plt.imshow(org_img)
    # plt.axis('off')
    # plt.show()
    prediction = model.predict(sharp_image)


    # Make prediction
    # try:
    #     prediction = model.predict(sharp_image)
    # except:
    #     prediction = model.predict(sharp_image.reshape(1,7*7*512*6))


    return prediction[0]
classes = ['Acinetobacter.baumanii',
           'Bacteroides.fragilis',
           'Candida.albicans',
           'Escherichia.coli',
           'Staphylococcus.aureus']
# @app.route("/")
# def home():
# 	return "HELLO from vercel use flask"
@app.route('/uploads', methods=['POST'])
def upload_file():
    import numpy as np

    MODEL_PATH = './mobileNet_5classes9829(3)/mobileNet_5classes9829(3)'

    model = load_model(MODEL_PATH)
    if request.method == 'POST':
        for field, data in request.files.items():
            print('field:', field)
            print('filename:', data.filename)
            if data.filename:

                data.save(os.path.join(FOLDER, data.filename))
                global p
                basepath = os.path.dirname(__file__)

                file_path = os.path.join(
                    basepath ,FOLDER, secure_filename(data.filename))
                p = print(os.path.abspath(__file__))
                print(file_path)
                global c
                c = print(data.filename)
                p = prediction(file_path, model)

                result = str(classes[np.argmax(p)])
                print(result)
            return result
    return None


#     return result
# return None

# if __name__ == "__main__":
#     app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=1935))
