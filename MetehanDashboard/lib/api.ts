// Yusuf'un Dashboard/app.py'si icindeki FastAPI'ye (embedded thread, port 8000)
// konusuyoruz. app.py Dash UI'siyla AYNI process'te calisiyor -- yani bu
// dashboard'un veri gormesi icin Yusuf'un "python app.py"si arka planda
// calisiyor olmali (onun Dash arayuzunu -- 8050 portu -- kullanmasak bile).
// Detay icin README.md'ye bak.

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

import type { AircraftInfo, Alert, Flight, HistoryPoint, RouteInfo } from "./types";

async function getJson<T>(path: string, timeoutMs = 5000): Promise<T | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal: controller.signal });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchFlights(): Promise<Flight[]> {
  return (await getJson<Flight[]>("/api/flights")) ?? [];
}

export async function fetchAlerts(): Promise<Alert[]> {
  return (await getJson<Alert[]>("/api/alerts")) ?? [];
}

export async function fetchRoute(callsign: string): Promise<RouteInfo> {
  const trimmed = callsign.trim();
  if (!trimmed) return { found: false };
  return (await getJson<RouteInfo>(`/api/route/${encodeURIComponent(trimmed)}`)) ?? { found: false };
}

export async function fetchAircraftInfo(icao24: string): Promise<AircraftInfo> {
  return (await getJson<AircraftInfo>(`/api/aircraft_info/${encodeURIComponent(icao24)}`)) ?? { found: false };
}

export async function fetchHistory(icao24: string, hours = 1): Promise<HistoryPoint[]> {
  // /api/history internal hata durumunda {"error": "..."} DICT'i doner (array
  // degil) -- 200 status ile. Array.isArray kontrolu olmadan bu, TypeScript'i
  // yaniltip runtime'da .map() gibi cagrilarda patlar.
  const data = await getJson<HistoryPoint[] | { error: string }>(
    `/api/history/${encodeURIComponent(icao24)}?hours=${hours}`,
  );
  return Array.isArray(data) ? data : [];
}
