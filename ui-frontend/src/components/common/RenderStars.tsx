// components/common/RenderStars.tsx
import React from "react";
import type { Theme } from "../../theme/theme";
import { rgba } from "../../theme/theme";

export type RenderStarsProps = {
  theme: Theme;
  stars: number | null | undefined;
};

export function RenderStars(props: RenderStarsProps): JSX.Element | null {
  const { theme, stars } = props;

  // Guard against null/undefined/non-numeric
  if (typeof stars !== "number" || !Number.isFinite(stars)) {
    return null;
  }

  // Clamp into [0, 5]
  const clamped = Math.max(0, Math.min(5, stars));

  // For display text: 0.1 increments
  const precise = Math.round(clamped * 10) / 10; // e.g. 4.23 -> 4.2

  // For star shapes: nearest integer (0–5)
  const fullStars = Math.round(clamped);

  const filledColor = theme.accent;
  const emptyColor = rgba(theme.text3, 0.7);

  const totalStars = 5;
  const items: JSX.Element[] = [];

  for (let i = 0; i < totalStars; i += 1) {
    const filled = i < fullStars;

    items.push(
      <span
        key={i}
        style={{
          color: filled ? filledColor : emptyColor,
          fontSize: 14,
          lineHeight: 1,
        }}
      >
        {filled ? "★" : "☆"}
      </span>
    );
  }

  return (
    <div
      aria-label={`${precise.toFixed(1)} out of ${totalStars} stars`}
      style={{ display: "inline-flex", alignItems: "center", gap: 4, marginLeft: 4 }}
    >
      {/* Stars (coarse bucket) */}
      <div style={{ display: "inline-flex", gap: 2 }}>{items}</div>

      {/* Numeric label (actual precision) */}
      <span
        style={{
          fontSize: 12,
          color: theme.text3,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {precise.toFixed(1)}
      </span>
    </div>
  );
}
