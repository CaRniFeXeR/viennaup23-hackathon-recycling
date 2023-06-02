console.log("cluster_map.js")

// Initialize the map
// var map = L.map('map').setView([51.505, -0.09], 13);
var map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -5
        });
    var bounds = [[-26.5,-25], [1021.5,1023]];
    var image = L.imageOverlay('./static_content/background.png', bounds).addTo(map);

    var sol = L.latLng([ 145, 175.2 ]);
    L.marker(sol).addTo(map);
    map.setView( [70, 120], 1);