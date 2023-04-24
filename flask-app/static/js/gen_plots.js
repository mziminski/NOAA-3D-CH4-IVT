d3.json('http://127.0.0.1:8000/volume?itime=0', function(error, data) {
    if (error) return console.warn(error);

    var layout = {
        title: 'Volume',
        scene: {
            camera: {
                up: {x: 0, y: 0, z: 1},
                center: {x: 0, y: 0, z: 0},
                eye: {x: -1.25, y: -1.25, z: 1.0}
            }
        },
        width:800,
        height:800,
        font: {
            color: 'white',
        },
        paper_bgcolor: "rgba(0,0,0,0)",
    }

    Plotly.newPlot('volume-plt', data, layout);
});


d3.json('http://127.0.0.1:8000/sphere?itime=0&level=0', function(error, data) {
    if (error) return console.warn(error);

    var layout = {
        title: 'Sphere',
        scene: {
            xaxis: {
                ticks:'', 
                title:'', 
                showgrid:false, 
                showline:false, 
                zeroline:false, 
                showbackground:false, 
                showticklabels:false
            },
            yaxis: {
                ticks:'',
                title:'',
                showgrid:false,
                showline:false,
                zeroline:false,
                showbackground:false,
                showticklabels:false
            },
            zaxis: {
                ticks:'',
                title:'',
                showgrid:false,
                showline:false,
                zeroline:false,
                showbackground:false,
                showticklabels:false
            },
            camera: {eye: {
                x: 1.15, 
                y: -1.15, 
                z: 1.15
            }}, 
            // aspectratio: {x:1, y:1, z:},
        },
        width:800,
        height:800,
        font: {
            color: 'white',
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        text_color: "rgba(255,255,255,0)",
    }

    Plotly.newPlot('sphere-plt', data, layout);
});
