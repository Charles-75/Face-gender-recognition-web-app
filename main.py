from flask import Flask, render_template, request
from utils import getwidth
from model_pipeline import pipeline_model
import os

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = './static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/workflow', methods=["GET"])
def workflow():
    return render_template('workflow.html')


@app.route('/face_app', methods=["GET", "POST"])
def face_app():
    # Process POST request (form submit)
    if request.method == "POST":
        f = request.files['image']
        file_name = f.filename
        path = os.path.join(UPLOAD_FOLDER, file_name)
        f.save(path)
        width = getwidth(path)

        # Pipeline model
        pipeline_model(path, filename=file_name, color='bgr')

        return render_template('face_app.html', file_upload=True, img_name=file_name, width=width)

    return render_template('face_app.html', file_upload=False, img_name=None, width=300)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

