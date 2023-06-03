from flask import Flask, jsonify, send_file, send_from_directory, request
from PIL import Image
import flask
import os
import json
from pathlib import Path

def load_data_from_jsons():
    script_dir = os.path.dirname(__file__)

    folder = Path(os.path.join(script_dir, "..", "data", "images_and_json_for_UI"))

    dict = {}

    class_id_to_label = {0: "can", 1 : "bottle",2: "glas"}

    for file in folder.iterdir():
        if file.name.startswith("pred_after_"):
            #load file as json
            with open(file) as f:
                json_data = json.load(f)
            dict[json_data["name"]] = [json_data["features"][0],json_data["features"][1], class_id_to_label[json_data["class_id"]]]

    return dict


app = Flask(__name__)


# example_data = {
#         "img17_0.png": [0.23939520415773685, 0.39729491531822647, "example_class"],
#         "img17_1.png": [80.7109474876517918, 1.5464037739352541],
#         "img17_2.png": [2.4823443230341764, 90.1366062742947145],
#         "img17_3.png": [3.8467880824049828, 3.7541917936743278],
#         "img17_4.png": [4.09798797908084622, 4.6638587418518253],
#         "img2_1.png": [0.06131955485075535, 80.7109474876517918],
#         "img2_2.png": [80.7109474876517918, 80.7109474876517918],
#         "img2_3.png": [0.015828088631563042, 130.7109474876517918],
#         "img2_4.png": [0.7671876772668083, 180.453705845175297],
#         "img3_0.png": [120.6255427512261426, 0.19949894591375994],
#         "img3_1.png": [0.5123464696557279, 0.21399429733929098],
#         "img3_2.png": [0.5718495766885707, 0.9615837987658215],
#         "img3_3.png": [0.6701364557109918, 0.2437979932943165],
#         "img3_4.png": [0.3927738101131796, 0.5888332234460126]
#     }


example_data = load_data_from_jsons()

objects_array = []
for img_name, point in example_data.items():
    img_obj = {
        'img_name': img_name,
        'point': [point[0], point[1]],
        'predicted_class': point[2] if len(point) > 2 else None
    }
    objects_array.append(img_obj)


@app.route('/get_latest_scans/<int:n>')
def get_latest_scans(n):
    if n <= 0:
        return jsonify({'error': 'Invalid value for n. Must be a positive integer.'}), 400

    returnArr = []

    counter = 0
    # for obj in objects_array:
    for counter in range(n):
        if(counter >= len(objects_array)):
            break

        img_url = "./static_content/scan_img/" + objects_array[counter]['img_name']
        img_size = [0, 0]
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "..", "data", "images", objects_array[counter]['img_name'])

        try:
            with Image.open(file_path) as img:
                img_size = img.size
        except IOError:
            counter += 1
            continue

        target_obj = {
            'id': counter,
            'point': objects_array[counter]['point'],
            'img_url': img_url,
            'img_size': img_size,
            'predicted_class': objects_array[counter]['predicted_class']
        }
        counter += 1

        returnArr.append(target_obj)

    return jsonify({'scanned_objects': returnArr})



@app.route('/set_object_class', methods=['POST'])
def set_object_class():
    data = request.get_json()

    if 'id' not in data or 'object_class' not in data:
        return jsonify({'error': 'Invalid request. Missing id or object_class.'}), 400

    object_id = data['id']
    object_class = data['object_class']

    if object_id < 0 or object_id >= len(objects_array):
        return jsonify({'error': 'Invalid object ID.'}), 400

    objects_array[object_id]['predicted_class'] = object_class

    return jsonify({'message': 'Object class updated successfully.'})

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
