console.log("cluster_map.js")

// Initialize the map
// var map = L.map('map').setView([51.505, -0.09], 13);

function genIcon(url, size, view_factor = 1.0) {

    scale_factor = 0.1;
    var bottleIcon = L.icon({
        iconUrl: url,
        iconSize: [size[0] * scale_factor * view_factor, size[1] * scale_factor * view_factor], // size of the icon
        init_iconSize: [size[0], size[1]], // size of the icon
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
    minZoom: 0
});

var bounds = [
    [-2500, -2500],
    [2500, 2500]
];
var objs = [];
var image = L.imageOverlay('./static_content/grid.png', bounds).addTo(map);



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
        var factor = 1.0

        if (currentZoom < 0) {
            factor = 10 / (10 + currentZoom);
        } else if (currentZoom > 0) {
            factor = 10 / (10 + currentZoom);
        }

        // Update the icon size of the marker
        marker.setIcon(genIcon(scan_obj.img_url, iconSize, factor));
    });


});

function getPopupContent(scan_obj) {
    var predicted_class = scan_obj.predicted_class !== null ? scan_obj.predicted_class : "";
    var popupContent = `
    <div class="custom-popup" id="obj_${scan_obj.id}">
        <img src="${scan_obj.img_url}" alt="Image" style="max-width: 100%; max-height: 100%;">
        <p>the object is labeled as <b>${predicted_class}</b></p>
        <input type="text" placeholder="type a new label" value="${predicted_class}">
        <button onclick="submitForm(${scan_obj.id})">Save</button>
    </div>
    `;

    return popupContent;
}


function addPoint(scan_obj) {
    // Create a custom popup content
    var popupContent = getPopupContent(scan_obj);

    var sol = L.latLng(scan_obj.point);
    var marker = L.marker(sol, { icon: genIcon(scan_obj.img_url, scan_obj.img_size, 1.0) }).addTo(map).bindPopup(popupContent);
    map.setView([70, 120], 1);
    scan_obj.marker = marker;
    objs.push(scan_obj);


}

// Custom JavaScript function for submitting the form
function submitForm(id) {
    var userInput = document.querySelector('#obj_' + id + ' input[type="text"]').value;

    var postData = {
        id: id,
        object_class: userInput
    };

    // Sende den POST-Request an den Backend-Endpunkt
    fetch('/set_object_class', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(postData)
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message); // Optional: Verarbeite die Antwort vom Backend
        })
        .catch(error => {
            console.error('Error:', error);
        });

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
                    console.log(scanObj);
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