from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
import os
import os.path

import numpy as np 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import joblib



app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
recogniton_model = load_model("D:\Documents\WesternAI\WesternAI\myModel1.h5")

class photoForm(FlaskForm):
    photo = FileField(validators=[FileRequired()])

def allowed_file(filename):
    print('hi')
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predictImg(img):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    img = image.load_img(img, grayscale=True, target_size=(48, 48))
    #show_img=image.load_img(img, grayscale=False, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = recogniton_model.predict(x)
  #emotion_analysis(custom[0])
    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    print(objects[ind])
    return objects[ind]



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = photoForm()
    if request.method == 'POST':
        f = form.photo.data
        if f and allowed_file(f.filename):
            extension = os.path.splitext(f.filename)[1]
            session['ext'] = extension
            filename = 'pre-predictImg' + extension #secure_filename(f.filename)
            path = 'D:\Documents\WesternAI\WesternAI\static\photos'
            f.save(os.path.join(
                path, filename
            ))
            return redirect(url_for('predict'))
    return render_template('westernai.html', form = form)

@app.route('/prediction')
def predict():
    img = 'D:\Documents\WesternAI\WesternAI\static\photos\pre-predictImg' + session['ext']
    prediction = predictImg(img)
    return render_template('prediction.html', value = prediction, img = img)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(debug=True)  

