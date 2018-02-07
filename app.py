
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise
import base64
from PIL import Image
from io import BytesIO
import topfivebreeds
import extract_feature_vector
import seaborn as sns
sns.set(font_scale=2)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
BREEDS_FOLDER = os.path.basename('breedscoreplots')
ALLOWED_EXTENSIONS = set(['png', 'gif', 'bmp', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BREEDS_FOLDER'] = BREEDS_FOLDER

featuredf = pd.read_pickle("pf_feature_vecs_red.pkl")
pfdf = pd.read_pickle("pfdfFULL.pkl")

pd.set_option('display.max_colwidth', -1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def maketop10df(top10list):
    simpetsdf = pd.DataFrame(columns=['Picture','Pet Name','Breeds','Shelter/Location/Contact','Similarity\\nScore'])
    for simpic in top10list.index:    
        petid,picnum = simpic.split('_')
        filestr = pfdf.loc[petid,'media'][int(picnum)-1][0]
        imgstr = '''<img style="border:3px solid white" src="'''+filestr+'''">'''
        simpetsdf.loc[simpic,'Picture'] = imgstr
        simpetsdf.loc[simpic,'Pet Name'] = pfdf.loc[petid,'petname']
        if pfdf.loc[petid,'breeds'][1]:
            simpetsdf.loc[simpic,'Breeds'] = ',\\n'.join(pfdf.loc[petid,'breeds'])
        else:
            simpetsdf.loc[simpic,'Breeds'] = pfdf.loc[petid,'breeds'][0]
        infodict={}
        for info in ['email','city','state','zip']:
            if pfdf.loc[petid,'contact'][info]:
                infodict[info]=pfdf.loc[petid,'contact'][info]['$t']   
            else:
                infodict[info] = ''
        if pfdf.notnull().loc[petid,'country']:
            infodict['country'] = pfdf.loc[petid,'country']
        else:
            infodict['country'] = ''
        if pfdf.notnull().loc[petid,'sheltername']:
            infodict['sheltername'] = pfdf.loc[petid,'sheltername']
        else:
            infodict['sheltername'] = ''
        simpetsdf.loc[simpic,'Shelter/Location/Contact'] = str(infodict['sheltername'] + '\\n'
                                                       + infodict['city']+', '
                                                       + infodict['state'] + ' ' 
                                                       + infodict['zip'] + ' '  
                                                       + infodict['country'] + '\\n'
                                                       + infodict['email'])

        simpetsdf.loc[simpic,'Similarity\\nScore'] = "{0:.4f}".format(top10list.loc[simpic])
    return simpetsdf


style = "<style> td{font-size: 18px;vertical-align: middle;text-align: center;} \
th{font-size: 21px; font-weight:bolder;}</style>"

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename=='':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
            file.save(f)
            if request.form['submit'] == 'Find similar pets':
                input_vector = extract_feature_vector.run_inference_on_image(f)
                #similarity = pd.Series(np.squeeze(pairwise.cosine_similarity(featuredf[feature_columns],input_vector.reshape(1,-1))), index=featuredf.index)
                similarity = pd.Series(np.squeeze(pairwise.cosine_similarity(featuredf,input_vector.reshape(1,-1))), index=featuredf.index)
                top10list = similarity.sort_values(ascending=False).head(10)
                simpetsdf = maketop10df(top10list)
                tablestr = style+simpetsdf.to_html(justify='center',escape=False,index=False).replace("\\n","<br>")
                return render_template('tableview.html',tablestr=tablestr, inputfile=filename)


    return render_template('index.html')


@app.route('/breedguesser', methods=['GET','POST'])
def breedguesser():
    if request.method == 'POST':
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename=='':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
            file.save(f)
            if request.form['submit'] == 'Classify breed':
                breeds, scores = topfivebreeds.top5graph(f)
                plotfilename = filename.split('.')[0]+'_breedscore.jpg'
                y_pos = range(len(breeds))
                plt.barh(y_pos[::1], scores[::-1], align='center', alpha=0.5)
                plt.yticks(y_pos[::1], breeds[::-1])
                plt.xlabel('Scores')
                plt.title('Breed scores')
                plt.tight_layout()
                f2 = os.path.join(app.config['BREEDS_FOLDER'], plotfilename)
                plt.savefig(f2)
                return render_template('breeddisplay.html', filename=filename, plotfilename=plotfilename)


    return render_template('breedguesser.html')

@app.route('/underconst', methods=['GET'])
def underconst():
     return render_template('underconst.html')


@app.route('/breedscoreplots/<filename>')
def breed_plot(filename):
    return send_from_directory(app.config['BREEDS_FOLDER'],
                               filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
