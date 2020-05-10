import os
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename, redirect
from src.MusicRegionPredictor import *

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'wav'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            secureFilename = secure_filename(f.filename)
            f.save(secureFilename)
            predictions1, predictions2 = predictSong(secureFilename)
            os.remove(secureFilename)
            return render_template(
                "result.html",
                type1Results=predictions1,
                type2Results=predictions2,
                length=4,
                fileName=f.filename
            )
        else:
            return "Incompatible file type. Only wav files are supported. Please Try again!"


if __name__ == '__main__':
    app.run(debug=True)

