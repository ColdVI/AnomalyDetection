"use client";

// Dashboard/app.py'deki _airplane_icon() ile ayni SVG -- ucus yonune
// (heading, 0-360, kuzeyden saat yonunde) gore CSS rotate uygulanmis
// bir ucak ikonu. Ikon 0 derecede kuzeye baktigi icin heading dogrudan
// rotate() acisi olarak kullanilabiliyor.
export function FlightIcon({
  heading,
  color,
  size = 22,
}: {
  heading: number;
  color: string;
  size?: number;
}) {
  return (
    <div
      style={{
        transform: `rotate(${heading}deg)`,
        transformOrigin: "center",
        width: size,
        height: size,
      }}
    >
      <svg width={size} height={size} viewBox="0 0 24 24">
        <path
          d="M12 1 L15 13 L23 18 L15 16 L15 20.5 L18.5 22.5 L12 21
             L5.5 22.5 L9 20.5 L9 16 L1 18 L9 13 Z"
          fill={color}
          stroke="#07070e"
          strokeWidth={0.5}
        />
      </svg>
    </div>
  );
}
