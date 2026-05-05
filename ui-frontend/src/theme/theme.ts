// src/theme/theme.ts

export type Theme = {
  bg: string;
  surface: string;
  surface2: string;
  surface3: string;
  border: string;
  border2: string;
  text: string;
  text2: string;
  text3: string;
  link: string;
  linkActive: string;
  accent: string;
  accent2: string;
  danger: string;
  onAccent: string; // Softer “white” used on blue gradients.
};

export const DARK_THEME: Theme = {
  // bg: "#070A12",      // very dark blue-black
  bg: "#04070E",         // darker page canvas

  surface: "#0E1116", // very dark blue-gray
  //surface: "#111623",    // optional, too light,  but helps hierarchy

  // surface2: "#0C0F14",// very dark blue-gray (slightly darker)
  surface2: "#131925",
  //surface2: "#141A26", // clearly lifted from bg

  // surface3: "#090C10",// near-black blue-gray
  surface3: "#171E2F",
  //surface3: "#1A2133", // raised / header feel

  border: "#1A2642",     // dark blue-gray
  border2: "#233459",    // dark blue-gray (slightly lighter)
  text: "#E5E7EB",       // very light neutral gray
  text2: "#D1D5DB",      // light neutral gray
  text3: "#9CA3AF",      // medium neutral gray
  link: "#C6D3F5",       // light blue
  linkActive: "#E3EBFF", // very light blue
  accent: "#2B5BB7",     // medium blue
  accent2: "#234B97",    // dark blue
  danger: "#ff6b6b",     // light red
  onAccent: "rgba(255,255,255,0.78)"
};

export const LIGHT_THEME: Theme = {
  bg: "#f6f7fb",         // very light blue-gray
  //surface: "#ffffff",    // white, too bright
  surface: "#fafafa",    // almost white
  surface2: "#f7f8fd",   // very light blue-gray
  surface3: "#eef1fb",   // light blue-gray
  border: "#d6dbea",     // light blue-gray
  border2: "#c9d0e5",    // light blue-gray (slightly darker)
  text: "#0f172a",       // very dark blue-gray
  text2: "#334155",      // dark blue-gray
  text3: "#64748b",      // medium blue-gray
  link: "#334155",       // dark blue-gray
  linkActive: "#0f172a", // very dark blue-gray
  accent: "#2B5BB7",     // medium blue
  accent2: "#234B97",    // dark blue
  danger: "#dc2626",     // medium red
  onAccent: "rgba(255,255,255,0.80)"
};

export function rgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const r = Number.parseInt(h.slice(0, 2), 16);
  const g = Number.parseInt(h.slice(2, 4), 16);
  const b = Number.parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, alpha)`.replace("alpha", String(alpha));
}
