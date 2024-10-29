from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import time
import numpy as np
import json


app = Flask(__name__)


class Model:
    """
    A class to load and manage TensorFlow models for gemstone classification.
    
    Attributes:
        name (str): Name of the model
        model (tf.keras.Model): Loaded TensorFlow model
        index (dict): Mapping of model output indices to class labels
    """
    def __init__(self, _name):
        """
        Initialize the Model class by loading a saved model and its index mapping.

        Args:
            _name (str): Name of the model to load

        Raises:
            Exception: If model files do not exist at the specified path
        """
        if os.path.exists(f"models/{_name}.keras"):
            self.name = _name
            self.model = tf.keras.models.load_model(f"models/{self.name}.keras")
            with open(f"models/{self.name}.shape", "r") as f:
                self.index = json.loads(f.read())
        else:
            raise(f"Model[{self.name}] does not exist. Please download the model from Github Page https://github.com/Samueli924/gemstone_classification")

    def predict(self, _image):
        """
        Make a prediction using the loaded model.

        Args:
            _image (tf.Tensor): Input image tensor

        Returns:
            str: Predicted class label
        """
        return self.index[np.argmax(self.model(_image))]


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """
    API endpoint to get information about loaded models.

    Returns:
        json: Dictionary containing names of loaded models for shape, color and identification
    """
    global shape_model, color_model, identification_model
    return jsonify({
        'shape': shape_model.name if shape_model else None,
        'color': color_model.name if color_model else None,
        'identification': identification_model.name if identification_model else None,
    })


@app.route('/', methods=['GET', 'POST'])
def predict():
    """
    Main endpoint for gemstone classification.
    
    GET: Returns the index page
    POST: Accepts an image and returns classification results
    
    Returns:
        GET: HTML template
        POST: JSON with classification results for shape, color and identification
    """
    _shape = None
    _color = None
    _identification = None

    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        _path = f'uploads/{str(int(time.time()))}.jpg'
        file.save(_path)

        _image = tf.io.read_file(_path)
        _image = tf.image.decode_jpeg(_image, channels=3)
        _image = tf.image.resize_with_pad(_image, 200, 200)
        _image = tf.expand_dims(_image, axis=0)
        if shape_model:
            _shape = shape_model.predict(_image)
        if color_model:
            _color = color_model.predict(_image)
        if identification_model:
            _identification = identification_model.predict(_image)

        result = {
            'shape': _shape if _shape else None,
            'color': _color if _color else None,
            'identification': _identification if _identification else None,
        }

        return jsonify(result)
    

if __name__ == '__main__':
    # Initialize models with specific versions
    # If model loading fails, set to None
    try:
        shape_model = Model("s.83k.custom.20241025")
    except:
        shape_model = None
    try:
        color_model = Model("c.83k.custom.20241025")
    except:
        color_model = None
    try:
        identification_model = Model("i.84k.custom.20241025")
    except:
        identification_model = None
    print(shape_model)
    app.run(debug=True)