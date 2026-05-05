// components/common/rightJustifiedMetric.tsx
import React from "react";
import type { Theme } from "../../theme/theme";

export function rightJustifiedMetric(
  theme: Theme,
  content: React.ReactNode,
  minWidth = 88
): JSX.Element {
  return (
    <code
      style={{
        color: theme.text2,
        display: "inline-block",
        minWidth,
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {content}
    </code>
  );
}
