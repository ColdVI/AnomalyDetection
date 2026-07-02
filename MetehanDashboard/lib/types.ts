// Bu tipler Dashboard/app.py (Yusuf'un FastAPI backend'i) ve
// Dashboard/KAFKA_SCHEMA.md'deki adsb.flights semasiyla BIREBIR eslesir.
// Backend degismedigi surece bu dosyaya dokunmaya gerek yok.

export type Flight = {
  icao24: string;
  callsign: string;
  lat: number;
  lon: number;
  alt: number; // metre (barometrik)
  velocity: number | null; // m/s
  track: number | null; // derece, 0-360, kuzeyden saat yonunde
  vertical_rate: number | null; // m/s
  category: string;
  squawk: string;
  emergency: string; // "none" | "general" | "lifeguard" | "minfuel" | "nordo" | "unlawful" | "downed" | "reserved"
  source: string;
  ts: string; // ISO 8601 UTC
};

export type Alert = {
  icao24: string;
  alert_type: string;
  score?: number;
  ts?: string;
};

export type RouteInfo = {
  found: boolean;
  airline?: string | null;
  origin_name?: string | null;
  origin_iata?: string | null;
  origin_city?: string | null;
  origin_lat?: number | null;
  origin_lon?: number | null;
  dest_name?: string | null;
  dest_iata?: string | null;
  dest_city?: string | null;
  dest_lat?: number | null;
  dest_lon?: number | null;
};

export type AircraftInfo = {
  found: boolean;
  type?: string;
  manufacturer?: string;
  registration?: string;
  owner?: string;
  owner_country?: string;
  photo_thumb?: string | null;
};

export type HistoryPoint = {
  _time: string;
  lat: number | null;
  lon: number | null;
  alt: number | null;
  velocity: number | null;
};
