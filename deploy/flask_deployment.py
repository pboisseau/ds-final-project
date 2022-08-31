from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def temp():
    return render_template('template.html')

@app.route('/',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = filename.split('.')[0] + '.csv'
            file.save(os.path.join('C:/Users/phili/OneDrive/Desktop/DataScience/ds-final-project/input/', new_filename))
            

        df = pd.read_csv('C:/Users/phili/OneDrive/Desktop/DataScience/ds-final-project/input/' + new_filename)
        
        file = 'C:/Users/phili/OneDrive/Desktop/DataScience/ds-final-project/FF_model.h5'
        
        loaded_model = tf.keras.models.load_model(file)
        model = loaded_model
        
        X_fit = StandardScaler().fit_transform(df)

        val_predz = model.predict(X_fit).flatten()
        
        df['predicted_points'] = val_predz
        df = df.sort_values(by=['predicted_points'],ascending=False)
        
        df.to_csv('C:/Users/phili/OneDrive/Desktop/DataScience/ds-final-project/output/results.csv')

        return render_template('processed.html')

    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)