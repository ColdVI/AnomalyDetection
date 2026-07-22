window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                if (!window.__aircraftCanvasRenderer) {
                    window.__aircraftCanvasRenderer = L.canvas({
                        padding: 0.5
                    });
                }
                const p = feature.properties;
                const color = p.color || '#00b4d8';
                const opacity = (p.opacity === undefined || p.opacity === null) ? 1 : p.opacity;
                return {
                    fillColor: color,
                    fillOpacity: opacity,
                    color: '#07070e',
                    weight: 0.6,
                    opacity: opacity,
                    renderer: window.__aircraftCanvasRenderer,
                };
            }

            ,
        function1: function(feature, layer, context) {
            const p = feature.properties;
            if (!p.icao24) {
                return;
            } // guvenlik icin birakildi, artik hep dolu (kumeleme kalkti)
            let signalRow = '';
            if (p.signal_age_text) {
                signalRow = '<div style="grid-column: 1 / -1;"><span style="color:#666">' +
                    p.lbl_signal_age + ' </span><span style="color:#f7b731">' + p.signal_age_text + '</span></div>';
            }
            const html = '<div style="min-width:150px">' +
                '<div style="font-size:14px; font-weight:700; color:' + p.color +
                '; margin-bottom:1px;">' + p.callsign + '</div>' +
                '<div style="font-size:10px; color:#888; margin-bottom:6px;">' + p.subtitle + '</div>' +
                '<div style="display:grid; grid-template-columns:1fr 1fr; gap:3px 12px; font-size:11px;">' +
                '<div><span style="color:#666">' + p.lbl_alt + ' </span><span>' + p.alt_text + '</span></div>' +
                '<div><span style="color:#666">' + p.lbl_speed + ' </span><span>' + p.speed_text + '</span></div>' +
                '<div><span style="color:#666">' + p.lbl_track + ' </span><span>' + p.track_text + '</span></div>' +
                '<div><span style="color:#666">' + p.lbl_vspeed + ' </span><span>' + p.vspeed_text + '</span></div>' +
                signalRow +
                '</div></div>';
            layer.bindTooltip(html, {
                direction: 'top',
                offset: [0, -14]
            });
        }

    }
});