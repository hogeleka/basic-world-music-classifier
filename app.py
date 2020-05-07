from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename, redirect
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
from MusicRegionPredictor import *

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload')
def upload():
    return render_template('index.html')
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            f.save(secure_filename(f.filename))
            # predictor = MusicRegionPredictor()
            predictedClass = predictRegion(f.filename)
            os.remove(f.filename)
            return f'file uploaded successfully. Predicted {predictedClass}'
        else:
            return "incompatible file type. go back and try again"
if __name__ == '__main__':
    app.run()