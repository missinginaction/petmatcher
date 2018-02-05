
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import breed_classify
import extract_feature_vector
import pandas as pd
from sklearn.metrics import pairwise
import base64
from PIL import Image
from io import BytesIO


app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
BREEDS_FOLDER = os.path.basename('breedscoreplots')
ALLOWED_EXTENSIONS = set(['png', 'gif', 'bmp', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BREEDS_FOLDER'] = BREEDS_FOLDER

featuredf = pd.read_pickle("pf_feature_vecs.pkl")
pfdf = pd.read_pickle("petsdf.pkl")
PETPHOTOPATH="/Users/miaCDIPS/petphotos/"

feature_columns = featuredf.columns
pd.set_option('display.max_colwidth', -1)

for ind in featuredf.index:
    petid = ind.split('_')[0]
    featuredf.loc[ind,'breed'] = pfdf.loc[petid,'breeds'][0]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((300, 1000), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'''<img src="data:image/jpeg;base64,{image_base64(im)}">'''

def maketop10df(top10list):
    simpetsdf = pd.DataFrame(columns=['file','name','breeds','shelter/location/contact'])
    for simpic in top10list:    
        petid = simpic.split('_')[0]
        picnum = simpic.split('_')[1]
        #filestr = PETPHOTOPATH+simpic+".jpg"
        #filestr = pfdf.loc[petid,'media'][int(picnum)-1][0]
        #imgstr = '<img src="'+filestr+'">'
        #simpetsdf.loc[simpic,'file'] = imgstr
        simpetsdf.loc[simpic,'file'] = PETPHOTOPATH+simpic+".jpg"
        simpetsdf.loc[simpic,'name'] = pfdf.loc[petid,'name']
        if pfdf.loc[petid,'breeds'][1]:
            simpetsdf.loc[simpic,'breeds'] = ',\\n'.join(pfdf.loc[petid,'breeds'])
        else:
            simpetsdf.loc[simpic,'breeds'] = pfdf.loc[petid,'breeds'][0]
        simpetsdf.loc[simpic,'shelter/location/contact'] = str(pfdf.loc[petid,'shelterId'] + '\\n'
                                                       +pfdf.loc[petid,'contact']['city']['$t']+', '
                                                       +pfdf.loc[petid,'contact']['state']['$t'] + '\\n'
                                                       +pfdf.loc[petid,'contact']['email']['$t'])
        #simpetsdf.loc[simpic,'similarity score'] = similarity.loc[simpic]
    return simpetsdf

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename=='':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
            file.save(f)
            if request.form['submit'] == 'Classify breed':
                plotstr = breed_classify.top5graph(filename,'uploads')
                return redirect(url_for('breed_plot',filename=plotstr))
            elif request.form['submit'] == 'Find similar pets':
                input_vector = extract_feature_vector.run_inference_on_image('uploads/'+filename)
                similarity = pd.Series(np.squeeze(pairwise.cosine_similarity(featuredf[feature_columns],input_vector.reshape(1,-1))), index=featuredf.index)
                top10list = similarity.sort_values(ascending=False).head(10).index
                simpetsdf = maketop10df(top10list)
                style = "<style> td{font-size: 16px;}</style>"
                #tablestr = style+simpetsdf.to_html(formatters={'file': image_formatter}, escape=False,index=False).replace("\\n","<br>")
                #tablestr = simpetsdf.to_html(formatters={'file': image_formatter}, escape=False,index=False).replace("\\n","<br>")
                tablestr = simpetsdf.to_html(formatters={'file': image_formatter}, escape=False,index=False).replace("\\n","<br>")
        
                return render_template('tableview.html',tablestr=tablestr)


    return render_template('index.html')


@app.route('/breedscoreplots/<filename>')
def breed_plot(filename):
    return send_from_directory(app.config['BREEDS_FOLDER'],
                               filename)
