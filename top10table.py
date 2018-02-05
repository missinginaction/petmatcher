import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from sklearn.metrics import pairwise


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
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def maketop10df(top10list):
    simpetsdf = pd.DataFrame(columns=['file','name','breeds','shelter/location/contact','similarity score'])
    for simpic in top10list:    
        petid = simpic.split('_')[0]
        simpetsdf.loc[simpic,'file'] = PETPHOTOPATH+simpic+'.jpg'
        simpetsdf.loc[simpic,'name'] = pfdf.loc[petid,'name']
        if pfdf.loc[petid,'breeds'][1]:
            simpetsdf.loc[simpic,'breeds'] = ',\\n'.join(pfdf.loc[petid,'breeds'])
        else:
            simpetsdf.loc[simpic,'breeds'] = pfdf.loc[petid,'breeds'][0]
        simpetsdf.loc[simpic,'shelter/location/contact'] = str(pfdf.loc[petid,'shelterId'] + '\\n'
                                                       +pfdf.loc[petid,'contact']['city']['$t']+', '
                                                       +pfdf.loc[petid,'contact']['state']['$t'] + '\\n'
                                                       +pfdf.loc[petid,'contact']['email']['$t'])
        simpetsdf.loc[simpic,'similarity score'] = similarity.loc[simpic]
    return simpetsdf



top10list = similarity.sort_values(ascending=False).head(10).index
simpetsdf = maketop10df(top10list)
display(inputimg)
style = "<style> td{font-size: 16px;}</style>"
HTML(style+simpetsdf.to_html(formatters={'file': image_formatter}, escape=False,index=False).replace("\\n","<br>"))
