from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename, redirect
from MusicRegionPredictor import *

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def getPredictionResultString(prediction):
#     # result = ""
#     # for i in range(len(prediction)):
#     #     result = result + str(i+1) + ". " + prediction[i] + "\n"
#     # return result
#     return [prediction[]]


@app.route('/')
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
            predictions1, predictions2 = predictSong(secure_filename(f.filename))
            os.remove(secure_filename(f.filename))
            return render_template(
                "uploaded_file_results.html",
                type1Results=predictions1,
                type2Results=predictions2,
                length=4
                # f'<p> Predictions: <br> {getPredictionResultString(predictions1)} </p>' \
                # f'<p> Predictions: <br> {getPredictionResultString(predictions2)} </p>'
            )
            # return f'<p> Predictions: <br> {getPredictionResultString(predictions1)} </p>' \
            #        f'<p> Predictions: <br> {getPredictionResultString(predictions2)} </p>'
        else:
            return "Incompatible file type. Try again :/"
    # test = ["String 1", "String 2", "String 3", "String 4"]
    # return render_template("uploaded_file_results.html", type1Results=test, type2Results=test, length=4)

if __name__ == '__main__':
    app.run(debug=True)

