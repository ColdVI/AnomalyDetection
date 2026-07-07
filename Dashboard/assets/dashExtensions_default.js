window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
                const p = feature.properties;
                const heading = p.track || 0;
                const color = p.color || '#00b4d8';
                const opacity = (p.opacity === undefined || p.opacity === null) ? 1 : p.opacity;
                const html = '<div style="transform: rotate(' + heading + 'deg); ' +
                    'transform-origin: center; width: 22px; height: 22px; opacity: ' + opacity + ';">' +
                    '<svg width="22" height="22" viewBox="0 0 24 24">' +
                    '<path d="M12 1 L15 13 L23 18 L15 16 L15 20.5 L18.5 22.5 L12 21 ' +
                    'L5.5 22.5 L9 20.5 L9 16 L1 18 L9 13 Z" fill="' + color + '" ' +
                    'stroke="#07070e" stroke-width="0.5"/></svg></div>';
                const icon = L.divIcon({
                    html: html,
                    className: '',
                    iconSize: [22, 22],
                    iconAnchor: [11, 11]
                });
                return L.marker(latlng, {
                    icon: icon
                });
            }

            ,
        function1: function(feature, layer, context) {
            const p = feature.properties;
            if (!p.icao24) {
                return;
            } // cluster balonu -- ucak degil, tooltip yok
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