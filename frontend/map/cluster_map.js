console.log("cluster_map.js")

// Initialize the map
// var map = L.map('map').setView([51.505, -0.09], 13);

function getIcon(url, size) {

    scale_factor = 0.2;
    var bottleIcon = L.icon({
        iconUrl: url,
        iconSize: [size[0] * scale_factor, size[1] * scale_factor], // size of the icon
        init_iconSize: [size[0] * scale_factor, size[1] * scale_factor], // size of the icon
        // shadowSize:   [50, 64], // size of the shadow
        // iconAnchor:   [22, 94], // point of the icon which will correspond to marker's location
        // shadowAnchor: [4, 62]  // the same for the shadow
        popupAnchor: [0, 0],
        iconAnchor: [0, 0]
    });
    return bottleIcon;
}


var map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -5
});

var bounds = [
    [-26.5, -25],
    [1021.5, 1023]
];
var objs = [];
// var image = L.imageOverlay('./static_content/background.png', bounds).addTo(map);

// Attach the 'zoomend' event to the map
map.on('zoomend', function() {
    var currentZoom = map.getZoom();
    console.log(currentZoom);

    //for each marker in markers
    //update the icon size
    objs.forEach(function(scan_obj) {
        var marker = scan_obj.marker;
        // Calculate the new size of the icon based on the current zoom level
        var iconSize = marker.getIcon().options.init_iconSize // Default size

        if (currentZoom < 0) {
            factor = 10 / (10 + currentZoom);
            iconSize = [iconSize[0] * factor, iconSize[1] * factor];
        } else if (currentZoom > 0) {
            factor = 10 / (10 + currentZoom);
            iconSize = [iconSize[0] * factor, iconSize[1] * factor];
        }

        // Update the icon size of the marker
        marker.setIcon(getIcon(scan_obj.img_url, iconSize));
    });


});

function getPopupContent(scan_obj) {
    var popupContent = `
    <div class="custom-popup" id="obj_${scan_obj.id}">
        <img src="${scan_obj.img_url}" alt="Image">
        <p>the object is labeled as <b>${scan_obj.predicted_class}</b></p>
        <input type="text" placeholder="type a new label" value="${scan_obj.predicted_class}">
        <button onclick="submitForm(${scan_obj.id})">Save</button>
    </div>
    `;

    return popupContent;
}


function addPoint(scan_obj) {
    // Create a custom popup content
    var popupContent = getPopupContent(scan_obj);

    var sol = L.latLng(scan_obj.point);
    var marker = L.marker(sol, { icon: getIcon(scan_obj.img_url, scan_obj.img_size) }).addTo(map).bindPopup(popupContent);
    map.setView([70, 120], 1);
    scan_obj.marker = marker;
    objs.push(scan_obj);


}

// Custom JavaScript function for submitting the form
function submitForm(id) {
    var userInput = document.querySelector('#obj_' + id + ' input[type="text"]').value;
    alert('You entered: ' + userInput);
    objs[id].predicted_class = userInput;

    objs[id].marker.getPopup().setContent(getPopupContent(objs[id]));
}

function getLatestScans(n) {
    if (n <= 0) {
        console.error('Invalid value for n. Must be a positive integer.');
        return;
    }

    fetch(`/get_latest_scans/${n}`)
        .then(response => response.json())
        .then(data => {
            if (data.hasOwnProperty('scanned_objects')) {
                const scannedObjects = data.scanned_objects;
                scannedObjects.forEach(scanObj => {
                    addPoint(scanObj);
                });
            } else if (data.hasOwnProperty('error')) {
                console.error(data.error);
            } else {
                console.error('Invalid response format.');
            }
        })
        .catch(error => {
            console.error('An error occurred:', error);
        });
}


getLatestScans(10)


addPoint({ 'id': 0, 'img_url': "./static_content/bottle1.png", "point": [145, 175.2], "predicted_class": "plastic", "img_size": [245, 355] })
addPoint({ 'id': 1, 'img_url': "./static_content/bottle1.png", "point": [120, 22], "predicted_class": "", "img_size": [245, 355] })