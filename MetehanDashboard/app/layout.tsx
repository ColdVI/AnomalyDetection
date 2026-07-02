import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Metehan Dashboard -- ADS-B Canli",
  description: "adsb.lol realtime ucus takibi -- MapLibre GL tabanli bireysel dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr" className="dark">
      <body>{children}</body>
    </html>
  );
}
