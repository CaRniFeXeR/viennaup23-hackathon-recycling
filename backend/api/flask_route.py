from flask import Flask, jsonify, send_file, send_from_directory
from PIL import Image
import flask
import os

app = Flask(__name__)

# Mock data for scanned objects
# scanned_objects = [
#     {
#         'embedding_vec': [0.1, 0.2, 0.3],
#         'point': [10, 20],
#         'img_url': 'https://example.com/image1.jpg',
#         'predicted_class': 'cat'
#     },
#     {
#         'embedding_vec': [0.4, 0.5, 0.6],
#         'point': [30, 40],
#         'img_url': 'https://example.com/image2.jpg',
#         'predicted_class': 'dog'
#     },
#     # Add more scanned objects as needed
# ]

example_data = {
        "img0_0.png": [0.23939520415773685, 0.39729491531822647],
        "img0_1.png": [0.7109474876517918, 0.5464037739352541],
        "img0_2.png": [0.4823443230341764, 0.1366062742947145],
        "img0_3.png": [0.8467880824049828, 0.7541917936743278],
        "img2_0.png": [0.09798797908084622, 0.6638587418518253],
        "img2_1.png": [0.06131955485075535, 0.3103182742510159],
        "img2_2.png": [0.09068516802941429, 0.47123703536053474],
        "img2_3.png": [0.015828088631563042, 0.7393210213742311],
        "img2_4.png": [0.7671876772668083, 0.453705845175297],
        "img3_0.png": [0.6255427512261426, 0.19949894591375994],
        "img3_1.png": [0.5123464696557279, 0.21399429733929098],
        "img3_2.png": [0.5718495766885707, 0.9615837987658215],
        "img3_3.png": [0.6701364557109918, 0.2437979932943165],
        "img3_4.png": [0.3927738101131796, 0.5888332234460126]
    }


@app.route('/get_latest_scans/<int:n>')
def get_latest_scans(n):
    if n <= 0:
        return jsonify({'error': 'Invalid value for n. Must be a positive integer.'}), 400

    TargetArr = []

    counter = 0
    for img_name, point in example_data.items():
        if(counter >= n):
            break
        
        counter += 1
        x, y = point
        img_url = "./static_content/" + img_name
        img_size = [0, 0]
        script_dir = os.path.dirname(__file__)
        #parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
        file_path = os.path.join(script_dir, "..", "segmentation", "samples", img_name)

        try:
            with Image.open(file_path) as img:
                img_size = img.size
        except IOError:
            continue


        target_obj = {
            'id': counter,
            'point': [x, y],
            'img_url': img_url,
            'img_size': img_size
        }

        TargetArr.append(target_obj)

    return jsonify({'scanned_objects': TargetArr})


@app.route('/get_image/<path:img_path>')
def get_image(img_path):
    # Here, you would typically fetch the image based on the provided img_path from your storage or database.
    # For the purpose of this example, we will return a placeholder image.
    placeholder_image_path = 'placeholder.jpg'
    return send_file(placeholder_image_path, mimetype='image/jpeg')

@app.route("/static_content/<path:filename>")
def return_js(filename: str):
    return send_from_directory("./../../frontend/map", filename)

@app.route("/")
def index():
    """Displays the index page accessible at '/'"""
    return flask.send_file("./../../frontend/map/index.html")

if __name__ == '__main__':
    app.run()
