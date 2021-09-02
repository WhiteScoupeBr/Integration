from flask import Flask
from flask import request
from flask import render_template
import json
import os
from datetime import datetime
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time
from datetime import datetime
import threading
import atexit

app = Flask(__name__)

def cgne(vector):

    path = pathlib.Path(__file__).parent.absolute()

    vector = np.matrix(vector)

    image = np.zeros((3600, 1))

    print('Importando matriz...')
    matrix_lines = pd.read_csv(str(path) + '\H-1.txt', sep=',', lineterminator='\n', header=None)
    print('Importação concluída')
    matrix = np.matrix(matrix_lines.to_numpy())
    print('matriz convertida:')

    r = vector - (matrix * image)
    p = matrix.T * r

    tx_erro = 0

    count = 0
    # while count < 5:
    now = datetime.now()
    dt_string = now.strftime("Data de Ínicio: %d/%m/%Y %H:%M:%S")
    start = time.time() 
    print("Data de Ínicio:", dt_string)	


    while tx_erro < float('1e-4'):

        print('i = ' + str(count))

        alpha = ((r.T * r) / (p.T * p)).item((0, 0))

        print('alpha: ' + str(alpha))

        next_image = image + (alpha * p)
        next_r = r - (alpha * (matrix * p))

        beta = ((next_r.T * next_r) / (r.T * r)).item((0, 0))

        p = matrix.T * next_r + beta * p
        image = next_image
        tx_erro = np.linalg.norm(next_r, 2) - np.linalg.norm(r, 2)
        print('tx_erro: ', tx_erro)
        r = next_r 

        count += 1
        
    print('\n')
    end = time.time()
    print("Tempo CGNE:")
    print(end - start," secs")
    print("Iterações: ", count)

    now = datetime.now()
    dt_string = now.strftime(" %d/%m/%Y %H:%M:%S")
    print("Data de Término:", dt_string)

    image = image.reshape(60, 60)

    image = (image - np.min(image))/np.ptp(image)
    png = plt.imsave('image.png', image, cmap='gray')



@app.route("/")
def hello_world():
    return render_template("index.html")



@app.route("/enviar_dados", methods=['POST'])
def enviar_dados():
    if request.method == 'POST':
        item = request.form.to_dict()
        with open('data.json','w') as outfile:
            json.dump(item,outfile)
        convert_all()
        return 'thankyou'


def convert_all():
    with open('data.json') as json_file:
        data = json.load(json_file)

    real =[]
    real_aux = []
    str_aux =''
    count = 1
    tam = len(data['vetor'])
    for x in data['vetor']:
        if x !=',':
            str_aux +=x
        else:
            real_aux.append(float(str_aux))
            real.append(real_aux)
            real_aux = []
            str_aux =''
        if(count == tam ):
            real_aux.append(float(str_aux))
            real.append(real_aux)
        count+=1

    cgne(real)