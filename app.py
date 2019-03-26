
## updated 24 March 2019 for tensorflow 1.14 -cmi

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import os
import glob
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import export_png

import pandas as pd

from sklearn.metrics import pairwise
import topfivebreeds
import extract_feature_vector
import math
#import seaborn as sns
#sns.set(font_scale=2)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
BREEDS_FOLDER = os.path.basename('breedscoreplots')
ALLOWED_EXTENSIONS = set(['png', 'gif', 'bmp', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BREEDS_FOLDER'] = BREEDS_FOLDER

#featuredf = pd.read_pickle("pf_feature_vecs_0.pkl")
pfdf = pd.read_pickle("petsdf_lean.pkl")
shelterdf = pd.read_pickle("sheltersdf.pkl")
featurevec_table_list = glob.glob("pf_feature_vecs_*.pkl") #containing 2048 dimensional feature vectors, 20k per table
num_tables_to_pick = 5  #to speed up, only select some of the tables in featurevec_table_list

pd.set_option('display.max_colwidth', -1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def maketop10df(top10list):
    simpetsdf = pd.DataFrame(columns=['Picture','Pet Name','Breeds','Shelter/Location/Contact','Percent\\nMatch'])
    petid_set = set()
    for simpic in top10list.index:    
        petid,picnum = simpic.split('_')
        if petid not in petid_set:
            petid_set.add(petid)
            filestr = pfdf.loc[petid,'media'][int(picnum)-1][0]
            imgstr = '''<img style="border:3px solid white" src="'''+filestr+'''">'''
            simpetsdf.loc[simpic,'Picture'] = imgstr
            simpetsdf.loc[simpic,'Pet Name'] = pfdf.loc[petid,'name']
            if isinstance(pfdf.loc[petid,'breeds'], tuple):
                simpetsdf.loc[simpic,'Breeds'] = ',\\n'.join(pfdf.loc[petid,'breeds'])
            else:
                simpetsdf.loc[simpic,'Breeds'] = pfdf.loc[petid,'breeds']
            infodict=defaultdict(str)
            if pfdf.loc[petid,'shelterId'] in shelterdf.index:
                shelt_id = pfdf.loc[petid,'shelterId']
                for info in ['email','city','state','zip', 'country', 'name' ]:
                    if shelterdf.notnull().loc[shelt_id,info]:
                        infodict[info]=shelterdf.loc[shelt_id,info]  
                
            simpetsdf.loc[simpic,'Shelter/Location/Contact'] = str(infodict['name'] + '\\n'
                                                        + infodict['city']+', '
                                                        + infodict['state'] + ' ' 
                                                        + infodict['zip'] + ' '  
                                                        + infodict['country'] + '\\n'
                                                        + infodict['email'])
    
            simpetsdf.loc[simpic,'Percent\\nMatch'] = "{0:.2f}".format(top10list.loc[simpic]*100) + "%"
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
                input_vector = extract_feature_vector.run_inference_on_image(f).astype(np.float32)
                top_lists = []
                for featurevec_table in random.sample(featurevec_table_list, num_tables_to_pick):
                	featuredf = pd.read_pickle(featurevec_table)
                	sim_frame = pd.DataFrame(columns=['sim','petid'])
                	sim_frame['sim'] = pd.Series(np.squeeze(pairwise.cosine_similarity(featuredf,input_vector.reshape(1,-1))), index=featuredf.index)
                	sim_frame['petid'] = sim_frame.index.str[:-2]
                	top_lists.append(sim_frame.sort_values('sim', ascending=False).drop_duplicates('petid').head(10))

                
                top10 = pd.concat(top_lists).sort_values('sim',ascending=False).drop_duplicates('petid').head(10)['sim']
                simpetsdf = maketop10df(top10)
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
                plotfilename = filename.split('.')[0]+'_breedscore.png'
                p = figure(y_range=breeds[::-1], plot_height=500,title="Breed scores",toolbar_location=None, tools="")
                p.hbar(right = scores[::-1], y=breeds[::-1], height=0.9)
                p.ygrid.grid_line_color = None
                p.x_range.start = 0
                p.xaxis.axis_label = 'Scores'
                p.axis.major_label_text_font_size = "14pt"
                f2 = os.path.join(app.config['BREEDS_FOLDER'], plotfilename)
                export_png(p, filename=f2)
                return render_template('breeddisplay.html', filename=filename, plotfilename=plotfilename)

    return render_template('breedguesser.html')

"""@app.route('/breedguesser', methods=['GET','POST'])
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
                plt.close()
                return render_template('breeddisplay.html', filename=filename, plotfilename=plotfilename)


    return render_template('breedguesser.html')"""


@app.route('/underconst', methods=['GET'])
def underconst():
     return render_template('underconst.html')

@app.route('/moreinfo', methods=['GET'])
def moreinfo():
     return render_template('moreinfo.html')


@app.route('/breedscoreplots/<filename>')
def breed_plot(filename):
    return send_from_directory(app.config['BREEDS_FOLDER'],
                               filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
  app.run(host='0.0.0.0')
