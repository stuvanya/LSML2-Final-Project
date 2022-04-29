import torch
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
from utils import generate_caption
from caption_net import CaptionNet
import flask
from beheaded_inception3 import beheaded_inception_v3

restored_network = CaptionNet()
restored_network.load_state_dict(torch.load('network_result.bin'))
with open('word_to_index.pickle', 'rb') as f:
    word_to_index = pickle.load(f)
inception = beheaded_inception_v3().eval()

app = flask.Flask(__name__)

@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict_image():
    img_path = flask.request.files.get('image')
    if not img_path:
        return flask.render_template("error.html", error="No image provided!")

    try:
        image = plt.imread(img_path)
        image = resize(image, (299, 299))

        plt.imsave("static/images/img.png", image)
        output = [generate_caption(inception, restored_network, word_to_index, image, t=5.) for i in range(5)]
        return flask.render_template("result.html", output=output)
    except Exception:
        return flask.render_template("error.html", error="Some error with image!")
    
if __name__ == "__main__":
    app.run(debug=False)
