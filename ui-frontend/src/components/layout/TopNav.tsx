// TopNav.tsx
import React, { CSSProperties } from "react";

type PageKeyLiteral = "home" | "about" | "contact";
import type { Theme } from "../../theme/theme";

type TopNavProps = {
  theme: Theme;
  isDark: boolean;
  activePage: PageKeyLiteral;
  onPageChange: (page: PageKeyLiteral) => void;
  navLinkStyle: (theme: Theme, active: boolean) => CSSProperties;
};

function rgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const r = Number.parseInt(h.slice(0, 2), 16);
  const g = Number.parseInt(h.slice(2, 4), 16);
  const b = Number.parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, alpha)`.replace("alpha", String(alpha));
}

export function TopNav(props: TopNavProps): JSX.Element {
  const { theme, isDark, activePage, onPageChange, navLinkStyle } = props;

  const links: Array<{ key: PageKeyLiteral; label: string }> = [
    { key: "home", label: "Home" },
    { key: "about", label: "About" },
    { key: "contact", label: "Contact" }
  ];

  // Tweaks #3: bigger, jazzier “P” mark (script P + sparkle)
  const logoBoxSize = 30;

  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div
          aria-label="Predictly logo"
          style={{
            width: logoBoxSize,
            height: logoBoxSize,
            borderRadius: 11,
            background: `radial-gradient(circle at 30% 25%, ${rgba(
              "#ffffff",
              0.12
            )}, transparent 45%), linear-gradient(180deg, ${theme.accent}, ${theme.accent2})`,
            color: theme.onAccent,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 950,
            fontSize: 22,
            lineHeight: 1,
            position: "relative",
            boxShadow: `0 12px 28px ${rgba(theme.accent2, 0.16)}`,
            border: `1px solid ${rgba("#ffffff", isDark ? 0.12 : 0.08)}`
          }}
        >
          <span style={{ transform: "translateY(-0.5px)" }}>𝓟</span>
          <span
            aria-hidden="true"
            style={{
              position: "absolute",
              right: 6,
              top: 5,
              fontSize: 10,
              opacity: 0.85,
              textShadow: `0 2px 10px ${rgba("#000000", 0.35)}`
            }}
          >
            ✦
          </span>
        </div>
        <div style={{ fontWeight: 700, fontSize: 19, color: theme.text }}>Predictly</div>
      </div>

      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        {links.map((l) => (
          <a
            key={l.key}
            href="#"
            onClick={(e) => {
              e.preventDefault();
              onPageChange(l.key);
            }}
            style={navLinkStyle(theme, activePage === l.key)}
          >
            {l.label}
          </a>
        ))}
      </div>
    </div>
  );
}
