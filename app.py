from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import sklearn
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])

def index():
    if request.method == 'POST':
        pil_model = float(request.form['model'])
        if pil_model == 1:
            model = pickle.load(open('knn_pickle','rb'))
            
        elif pil_model == 2:
            model = pickle.load(open('dectree_pickle','rb'))      
        else:
            model = pickle.load(open('ann_pickle','rb'))
                    
        lahir = float(request.form['lahir']) 
        glukosa = float(request.form['glukosa'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        riwayat = float(request.form['riwayat'])
        umur = float(request.form['umur'])

        datas = np.array((lahir,glukosa,insulin,bmi,riwayat,umur))
        datas = np.reshape(datas, (1, -1))
        
        hasil = model.predict(datas)
        return render_template('hasil.html', final=hasil)

    
    else:
        return render_template('index.html')

if (__name__) == '__main__':
    app.run(debug=True)