import React from "react";

type Theme = {
  accent: string;
};

export function SpinnerGlobe(props: { theme: Theme }): JSX.Element {
  const { theme } = props;

  const size = 22;        // was 18
  const stroke = 2.5;    // was 2

  return (
    <span style={{ display: "inline-flex", alignItems: "center" }}>
      <style>
        {`
          @keyframes predictlySpin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }

          @media (prefers-color-scheme: dark) {
            .spinner-globe {
              filter:
                brightness(1.25)
                saturate(1.15)
                drop-shadow(0 0 1.5px rgba(80,160,255,0.45))
            }
          }
        `}
      </style>
      <svg
        className="spinner-globe"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        style={{
          animation: "predictlySpin 1s linear infinite",
          filter: "drop-shadow(0 0 0.5px rgba(0,0,0,0.2))"
        }}
        aria-label="Loading"
      >
        <circle
          cx="12"
          cy="12"
          r="9"
          fill="none"
          stroke={theme.accent}
          strokeWidth={stroke}
        />
        <path
          d="M3 12h18"
          fill="none"
          stroke={theme.accent}
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d="M12 3c3.5 3.5 3.5 14 0 18"
          fill="none"
          stroke={theme.accent}
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d="M12 3c-3.5 3.5-3.5 14 0 18"
          fill="none"
          stroke={theme.accent}
          strokeWidth={stroke}
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}
