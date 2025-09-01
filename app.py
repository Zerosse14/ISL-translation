import base64
import io
from PIL import Image
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/process_video_frame', methods=['POST'])
def process_video_frame():
    data = request.get_json()
    image_data = data['image']
    
    # Remove the prefix 'data:image/png;base64,' from the image string
    image_data = image_data.split(',')[1]
    
    # Decode the Base64 image
    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image))

    # Convert the image to a format suitable for your model (e.g., NumPy array)
    image = np.array(image)

    # Process the image with your AI model
    # For example, use your_model.predict(image) for inference
    # prediction = your_model.predict(image)

    # For now, just return a mock response
    return jsonify({'result': 'Frame received and processed successfully!'})

if __name__ == '__main__':
    app.run(debug=True)