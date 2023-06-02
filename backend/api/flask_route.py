from flask import Flask, jsonify, send_file, send_from_directory
import flask

app = Flask(__name__)

# Mock data for scanned objects
scanned_objects = [
    {
        'embedding_vec': [0.1, 0.2, 0.3],
        'point': [10, 20],
        'img_url': 'https://example.com/image1.jpg',
        'predicted_class': 'cat'
    },
    {
        'embedding_vec': [0.4, 0.5, 0.6],
        'point': [30, 40],
        'img_url': 'https://example.com/image2.jpg',
        'predicted_class': 'dog'
    },
    # Add more scanned objects as needed
]

@app.route('/get_latest_scans/<int:n>')
def get_latest_scans(n):
    if n <= 0:
        return jsonify({'error': 'Invalid value for n. Must be a positive integer.'}), 400
    if n > len(scanned_objects):
        return jsonify({'error': 'Requested more scans than available.'}), 400

    latest_scans = scanned_objects[-n:]
    return jsonify({'scanned_objects': latest_scans})


@app.route('/get_image/<path:img_path>')
def get_image(img_path):
    # Here, you would typically fetch the image based on the provided img_path from your storage or database.
    # For the purpose of this example, we will return a placeholder image.
    placeholder_image_path = 'placeholder.jpg'
    return send_file(placeholder_image_path, mimetype='image/jpeg')

@app.route("/static_content/<path:filename>")
def return_js(filename: str):
    # return flask.send_file("./../../frontend/map/cluster_map.js")
    return send_from_directory("./../../frontend/map", filename)

@app.route("/")
def index():
    """Displays the index page accessible at '/'"""
    return flask.send_file("./../../frontend/map/index.html")

if __name__ == '__main__':
    app.run()
