"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Feature, FeatureCollection, Point } from "geojson";

import {
  Map,
  MapArc,
  MapClusterLayer,
  MapControls,
  MapMarker,
  MapRoute,
  MarkerContent,
  MarkerLabel,
  MarkerTooltip,
} from "@/components/ui/mapcn-map-arc";
import { FlightIcon } from "@/components/flight-icon";
import { fetchAircraftInfo, fetchAlerts, fetchFlights, fetchHistory, fetchRoute } from "@/lib/api";
import { CATEGORY_LABELS, EMERGENCY_LABELS, POLL_INTERVAL_MS } from "@/lib/constants";
import type { AircraftInfo, Alert, Flight, RouteInfo } from "@/lib/types";

type FlightProps = {
  icao24: string;
  callsign: string;
  alt: number;
  velocity: number | null;
  track: number | null;
  isAlert: boolean;
};

const ISTANBUL_TZ = "Europe/Istanbul";
// Turkiye merkezli varsayilan gorunum -- adsb_producer.py'nin varsayilan
// poll noktasiyla (lat 39, lon 35) tutarli. Global olcege cikildiginda
// bu sadece baslangic viewport'u, MapClusterLayer zaten tum dunyayi
// gosterebilir.
const DEFAULT_CENTER: [number, number] = [35, 39];
const DEFAULT_ZOOM = 5;

function formatTime(iso: string | undefined): string {
  if (!iso) return "—";
  try {
    return new Intl.DateTimeFormat("tr-TR", {
      timeZone: ISTANBUL_TZ,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(new Date(iso));
  } catch {
    return "—";
  }
}

export function Dashboard() {
  const [flights, setFlights] = useState<Flight[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedIcao, setSelectedIcao] = useState<string | null>(null);
  const [route, setRoute] = useState<RouteInfo | null>(null);
  const [aircraftInfo, setAircraftInfo] = useState<AircraftInfo | null>(null);
  const [historyPositions, setHistoryPositions] = useState<[number, number][]>([]);
  // Cluster gorunumu: cok fazla ucak oldugunda (global olcek) performansli
  // kalsin diye acilabilir. Varsayilan kapali -- su anki bolgesel (Turkiye)
  // olcekte tek tek dondurulmus ikonlar (alert rengiyle) daha bilgilendirici.
  const [clusterMode, setClusterMode] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>("—");

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadFlightsAndAlerts = useCallback(async () => {
    const [f, a] = await Promise.all([fetchFlights(), fetchAlerts()]);
    setFlights(f);
    setAlerts(a);
    setLastUpdate(formatTime(new Date().toISOString()));
  }, []);

  useEffect(() => {
    loadFlightsAndAlerts();
    pollingRef.current = setInterval(loadFlightsAndAlerts, POLL_INTERVAL_MS);
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [loadFlightsAndAlerts]);

  const alertIcaoSet = useMemo(() => new Set(alerts.map((a) => a.icao24)), [alerts]);

  const selectedFlight = useMemo(
    () => flights.find((f) => f.icao24 === selectedIcao) ?? null,
    [flights, selectedIcao],
  );

  // Secim degisince rota + ucak bilgisi + son 1 saatlik gecmisi cek.
  // SADECE secim degisince -- her 15sn'lik tick'te tekrar sorgulanmaz
  // (app.py'deki update_route_info ile ayni tasarim gerekcesi: rota/tescil
  // bilgisi nadiren degisir, dis API'yi/cache'i gereksiz yormayalim).
  useEffect(() => {
    if (!selectedIcao) {
      setRoute(null);
      setAircraftInfo(null);
      setHistoryPositions([]);
      return;
    }
    let cancelled = false;

    const callsign = selectedFlight?.callsign?.trim();
    if (callsign) {
      fetchRoute(callsign).then((r) => {
        if (!cancelled) setRoute(r);
      });
    } else {
      setRoute({ found: false });
    }

    fetchAircraftInfo(selectedIcao).then((info) => {
      if (!cancelled) setAircraftInfo(info);
    });

    fetchHistory(selectedIcao, 1).then((points) => {
      if (cancelled) return;
      const positions: [number, number][] = points
        .filter((p) => p.lat != null && p.lon != null)
        .map((p) => [p.lon as number, p.lat as number]);
      setHistoryPositions(positions.length >= 2 ? positions : []);
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIcao]);

  // MapClusterLayer icin GeoJSON -- her ucak bir Point Feature, properties
  // icinde tum ihtiyacimiz olan alanlar (onPointClick geri donusunde bu
  // properties'e erisebiliyoruz, ekstra bir "flights" lookup'a gerek kalmaz).
  const clusterGeoJson = useMemo<FeatureCollection<Point, FlightProps>>(() => {
    return {
      type: "FeatureCollection",
      features: flights
        .filter((f) => Number.isFinite(f.lat) && Number.isFinite(f.lon))
        .map<Feature<Point, FlightProps>>((f) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [f.lon, f.lat] },
          properties: {
            icao24: f.icao24,
            callsign: f.callsign,
            alt: f.alt,
            velocity: f.velocity,
            track: f.track,
            isAlert: alertIcaoSet.has(f.icao24),
          },
        })),
    };
  }, [flights, alertIcaoSet]);

  const handleClusterPointClick = useCallback(
    (feature: Feature<Point, FlightProps>) => {
      setSelectedIcao(feature.properties.icao24);
    },
    [],
  );

  const canDrawArc =
    route?.found &&
    route.origin_lat != null &&
    route.origin_lon != null &&
    route.dest_lat != null &&
    route.dest_lon != null;

  const emergencyLabel =
    selectedFlight && EMERGENCY_LABELS[selectedFlight.emergency ?? "none"];
  const squawkIsEmergency =
    selectedFlight && ["7500", "7600", "7700"].includes(selectedFlight.squawk ?? "");

  return (
    <div className="relative h-screen w-full overflow-hidden bg-background">
      <Map center={DEFAULT_CENTER} zoom={DEFAULT_ZOOM} className="h-full w-full">
        <MapControls />

        {clusterMode ? (
          <MapClusterLayer<FlightProps>
            data={clusterGeoJson}
            pointColor="#00b4d8"
            onPointClick={handleClusterPointClick}
          />
        ) : (
          flights
            .filter((f) => Number.isFinite(f.lat) && Number.isFinite(f.lon))
            .map((f) => {
              const isAlert = alertIcaoSet.has(f.icao24);
              return (
                <MapMarker
                  key={f.icao24}
                  longitude={f.lon}
                  latitude={f.lat}
                  onClick={() => setSelectedIcao(f.icao24)}
                >
                  <MarkerContent>
                    <FlightIcon
                      heading={f.track ?? 0}
                      color={isAlert ? "#e63946" : "#00b4d8"}
                    />
                  </MarkerContent>
                  <MarkerTooltip>
                    {f.icao24.toUpperCase()} | {f.callsign?.trim() || "—"} | alt=
                    {f.alt.toFixed(0)}m
                    {f.velocity != null ? ` | ${f.velocity.toFixed(0)}m/s` : ""}
                  </MarkerTooltip>
                </MapMarker>
              );
            })
        )}

        {/* Secili ucagin son 1 saatlik gercek rotasi (InfluxDB'den) */}
        {historyPositions.length >= 2 && (
          <MapRoute coordinates={historyPositions} color="#00b4d8" width={3} opacity={0.6} />
        )}

        {/* Kalkis-varis havalimani arki -- SADECE backend origin/dest
            koordinatlarini donduruyorsa cizilir (bkz. README'deki opsiyonel
            app.py yamasi). Yoksa route bilgisi panelde metin olarak kalir,
            hata vermez. */}
        {canDrawArc && selectedIcao && (
          <MapArc
            data={[
              {
                id: selectedIcao,
                from: [route!.origin_lon as number, route!.origin_lat as number],
                to: [route!.dest_lon as number, route!.dest_lat as number],
              },
            ]}
            curvature={0.15}
            paint={{ "line-color": "#f77f00", "line-width": 2, "line-dasharray": [2, 2] }}
            interactive={false}
          />
        )}
      </Map>

      {/* --------------------------------------------------- Durum cubugu */}
      <div className="pointer-events-none absolute top-3 left-1/2 z-10 -translate-x-1/2 rounded-full bg-card/90 px-4 py-1.5 text-xs text-foreground shadow-md">
        {lastUpdate} | {flights.length} aktif uçuş | {alerts.length} alarm
        {clusterMode ? " | küme görünümü" : ""}
      </div>

      {/* -------------------------------------------- Gorunum degistirici */}
      <button
        type="button"
        onClick={() => setClusterMode((v) => !v)}
        className="absolute top-3 left-3 z-10 rounded-md border border-border bg-card/90 px-3 py-1.5 text-xs font-medium text-foreground shadow-md hover:bg-muted"
      >
        {clusterMode ? "Tekil görünüme geç" : "Küme görünümüne geç"}
      </button>

      {/* ----------------------------------------------------- Alarm paneli */}
      <div className="absolute top-3 right-3 z-10 max-h-56 w-64 overflow-y-auto rounded-lg bg-card/95 p-3 shadow-md">
        <h4 className="mb-2 text-xs font-semibold text-destructive">Model Alarmları</h4>
        {alerts.length === 0 ? (
          <div className="text-xs text-muted-foreground">Model henüz alarm üretmedi</div>
        ) : (
          alerts.map((a, i) => (
            <div
              key={`${a.icao24}-${i}`}
              className="mb-1 border-l-2 border-destructive py-1 pl-2 text-xs text-destructive"
            >
              🔴 {a.icao24} {a.alert_type}
            </div>
          ))
        )}
      </div>

      {/* -------------------------------------------------- Sol bilgi paneli */}
      <div
        className={`absolute top-0 bottom-0 left-0 z-20 w-80 overflow-y-auto bg-card/97 p-4 shadow-2xl transition-transform duration-300 ${
          selectedIcao ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-medium text-accent">Uçak Bilgisi</h3>
          <button
            type="button"
            onClick={() => setSelectedIcao(null)}
            aria-label="Kapat"
            className="text-2xl leading-none text-muted-foreground hover:text-foreground"
          >
            ×
          </button>
        </div>

        {selectedFlight ? (
          <>
            {(emergencyLabel || squawkIsEmergency) && (
              <div className="mb-2 rounded-md bg-destructive px-3 py-2 text-center text-sm font-semibold text-white">
                ⚠ {emergencyLabel ?? `ACİL DURUM SQUAWK: ${selectedFlight.squawk}`}
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              {[
                ["ICAO24", selectedFlight.icao24.toUpperCase()],
                ["Çağrı Kodu", selectedFlight.callsign?.trim() || "—"],
                ["Enlem", `${selectedFlight.lat.toFixed(4)}°`],
                ["Boylam", `${selectedFlight.lon.toFixed(4)}°`],
                ["İrtifa", `${selectedFlight.alt.toFixed(0)} m`],
                ["Hız", selectedFlight.velocity != null ? `${selectedFlight.velocity.toFixed(0)} m/s` : "—"],
                ["Yön", selectedFlight.track != null ? `${selectedFlight.track.toFixed(0)}°` : "—"],
                [
                  "Dikey Hız",
                  selectedFlight.vertical_rate != null
                    ? `${selectedFlight.vertical_rate > 0 ? "+" : ""}${selectedFlight.vertical_rate.toFixed(1)} m/s`
                    : "—",
                ],
                ["Kategori", CATEGORY_LABELS[selectedFlight.category] ?? selectedFlight.category ?? "—"],
                ["Squawk", selectedFlight.squawk || "—"],
                ["Son Güncelleme", formatTime(selectedFlight.ts)],
              ].map(([label, value]) => (
                <div key={label} className="rounded-md bg-muted px-2.5 py-2">
                  <div className="text-[11px] text-muted-foreground">{label}</div>
                  <div className="text-[15px] font-medium text-accent">{value}</div>
                </div>
              ))}
            </div>

            {route && (
              <div className="mt-3 rounded-md bg-muted p-2.5 text-[13px]">
                {!route.found ? (
                  <div className="text-muted-foreground">
                    Rota bilgisi bulunamadı (adsbdb.com veritabanında yok).
                  </div>
                ) : (
                  <>
                    {route.airline && (
                      <div className="mb-1 text-[11px] text-muted-foreground">✈ {route.airline}</div>
                    )}
                    <div>
                      <span className="text-accent">
                        {route.origin_city ?? "?"} ({route.origin_iata ?? "—"})
                      </span>
                      <span className="text-muted-foreground">  →  </span>
                      <span className="text-accent">
                        {route.dest_city ?? "?"} ({route.dest_iata ?? "—"})
                      </span>
                    </div>
                    {!canDrawArc && (
                      <div className="mt-1 text-[11px] text-muted-foreground">
                        (Haritada ok çizmek için backend&apos;e origin/dest koordinatı eklenmeli
                        — bkz. README.)
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {aircraftInfo?.found && (
              <div className="mt-3 rounded-md bg-muted p-2.5 text-[13px]">
                {(aircraftInfo.manufacturer || aircraftInfo.type) && (
                  <div className="font-medium text-accent">
                    {aircraftInfo.manufacturer} {aircraftInfo.type}
                  </div>
                )}
                {aircraftInfo.registration && <div>Tescil: {aircraftInfo.registration}</div>}
                {aircraftInfo.owner && (
                  <div>
                    {aircraftInfo.owner}
                    {aircraftInfo.owner_country ? ` (${aircraftInfo.owner_country})` : ""}
                  </div>
                )}
                {aircraftInfo.photo_thumb && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={aircraftInfo.photo_thumb}
                    alt="Uçak fotoğrafı"
                    className="mt-2 w-full rounded-md"
                  />
                )}
              </div>
            )}
          </>
        ) : (
          <div className="text-sm text-muted-foreground">
            {selectedIcao} şu anda sinyal göndermiyor (kapsama alanından çıkmış olabilir).
          </div>
        )}
      </div>
    </div>
  );
}
