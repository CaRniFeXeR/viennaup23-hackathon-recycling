console.log("cluster_map.js")

// Initialize the map
// var map = L.map('map').setView([51.505, -0.09], 13);

var bottleIcon = L.icon({
    iconUrl: './static_content/bottle1.png',
    iconSize:     [530, 355], // size of the icon
    shadowSize:   [50, 64], // size of the shadow
    iconAnchor:   [22, 94], // point of the icon which will correspond to marker's location
    shadowAnchor: [4, 62],  // the same for the shadow
    popupAnchor:  [-3, -76] // point from which the popup should open relative to the iconAnchor
});
var map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -5
    });

var bounds = [[-26.5,-25], [1021.5,1023]];
// var image = L.imageOverlay('./static_content/background.png', bounds).addTo(map);


// Create a custom popup content
var popupContent = `
<div class="custom-popup">
    <img src="./static_content/bottle1.png" alt="Image">
    <p>Some text goes here...</p>
    <input type="text" placeholder="Enter your text">
    <button onclick="submitForm()">Submit</button>
</div>
`;

var sol = L.latLng([ 145, 175.2 ]);
var marker = L.marker(sol, {icon:bottleIcon}).addTo(map).bindPopup(popupContent);
map.setView( [70, 120], 1);

// Attach the 'zoomend' event to the map
map.on('zoomend', function () {
    var currentZoom = map.getZoom();
    console.log(currentZoom);

    // Calculate the new size of the icon based on the current zoom level
    var iconSize = [530, 355]; // Default size
    // if (currentZoom > 10) {
    //     iconSize = [iconSize[0]*1.2, iconSize[1]*1.2]; // Increase size for zoom level greater than 10
    // } else if (currentZoom < 8) {
    //     iconSize = [iconSize[0]*0.8, iconSize[1]*0.8]; // Decrease size for zoom level less than 8
    // } else if (currentZoom < 7) {
    //     iconSize = [iconSize[0]*0.7, iconSize[1]*0.7]; // Decrease size for zoom level less than 6
    // } else if (currentZoom < 6) {
    //     iconSize = [iconSize[0]*0.6, iconSize[1]*0.6]; // Decrease size for zoom level less than 6
    // } else if (currentZoom < 5) {
    //     iconSize = [iconSize[0]*0.5, iconSize[1]*0.5]; // Decrease size for zoom level less than 6
    // }

    if (currentZoom < 0) {
        factor = 10/(10 + currentZoom);
        iconSize = [iconSize[0]*factor, iconSize[1]*factor];
    } else if (currentZoom > 0) {
        factor = 10/(10 + currentZoom);
        iconSize = [iconSize[0]*factor, iconSize[1]*factor];    
    }

    // Update the icon size of the marker
    marker.setIcon(L.icon({
        iconUrl: './static_content/bottle1.png',
        iconSize: iconSize
    }));
});


 // Custom JavaScript function for submitting the form
 function submitForm() {
    var userInput = document.querySelector('.custom-popup input[type="text"]').value;
    alert('You entered: ' + userInput);
}