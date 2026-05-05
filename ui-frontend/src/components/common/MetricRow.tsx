// components/common/MetricRow.tsx
import React from "react";
import type { Theme } from "../../theme/theme";
import { styles } from "../../theme/styles";
import { RichTooltip } from "./RichTooltip";

type MetricRowProps = {
  theme: Theme;
  label: string;
  value: React.ReactNode;
  tooltipHtml?: string;
  stars?: React.ReactNode;
};

export function MetricRow(props: MetricRowProps): JSX.Element {
  const { theme, label, value, tooltipHtml, stars } = props;

  return (
    <>
      <div
        style={{
          ...styles.formLabel(theme),
          // keep label on a single line; no ellipsis
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          flexWrap: "wrap",
        }}
      >
        {value}
        {stars ?? null}
        {tooltipHtml ? <RichTooltip html={tooltipHtml} theme={theme} /> : null}
      </div>
    </>
  );
}


// // components/common/MetricRow.tsx
// import React from "react";
// import type { Theme } from "../../theme/theme";
// import { styles } from "../../theme/styles";
// import { RichTooltip } from "./RichTooltip";

// type MetricRowProps = {
//   theme: Theme;
//   label: string;
//   value: React.ReactNode;
//   tooltipHtml?: string;
//   stars?: React.ReactNode;
// };

// export function MetricRow(props: MetricRowProps): JSX.Element {
//   const { theme, label, value, tooltipHtml, stars } = props;

//   return (
//     <>
//       <div
//         style={{
//           ...styles.formLabel(theme),
//           // keep label on a single line; no wrapping, no ellipsis
//           whiteSpace: "nowrap",
//         }}
//       >
//         {label}
//       </div>
//       <div
//         style={{
//           display: "flex",
//           alignItems: "center",
//           gap: 6,
//           flexWrap: "wrap",
//           justifyContent: "space-between", // Option B
//           width: "100%",                   // fill grid column
//         }}
//       >
//         {value}
//         {stars ?? null}
//         {tooltipHtml ? <RichTooltip html={tooltipHtml} theme={theme} /> : null}
//       </div>
//     </>
//   );
// }


// // components/common/MetricRow.tsx
// import React from "react";
// import type { Theme } from "../../theme/theme";
// import { styles } from "../../theme/styles";
// import { RichTooltip } from "./RichTooltip";

// type MetricRowProps = {
//   theme: Theme;
//   label: string;
//   value: React.ReactNode;
//   tooltipHtml?: string;
//   stars?: React.ReactNode;
// };

// export function MetricRow(props: MetricRowProps): JSX.Element {
//   const { theme, label, value, tooltipHtml, stars } = props;

//   return (
//     <>
//       <div
//         style={{
//           ...styles.formLabel(theme),
//           // keep label on a single line; no wrapping, no ellipsis
//           whiteSpace: "nowrap",
//         }}
//       >
//         {label}
//       </div>
//       <div
//         style={{
//           display: "flex",
//           alignItems: "center",
//           gap: 6,
//           flexWrap: "wrap",
//           justifyContent: "flex-end", // Option A
//           width: "100%",              // critical: fill grid column
//         }}
//       >
//         {value}
//         {stars ?? null}
//         {tooltipHtml ? <RichTooltip html={tooltipHtml} theme={theme} /> : null}
//       </div>
//     </>
//   );
// }


// // components/common/MetricRow.tsx
// import React from "react";
// import type { Theme } from "../../theme/theme";
// import { styles } from "../../theme/styles";
// import { RichTooltip } from "./RichTooltip";

// type MetricRowProps = {
//   theme: Theme;
//   label: string;
//   value: React.ReactNode;
//   tooltipHtml?: string;
//   stars?: React.ReactNode;
// };

// export function MetricRow(props: MetricRowProps): JSX.Element {
//   const { theme, label, value, tooltipHtml, stars } = props;

//   return (
//     <>
//       <div
//         style={{
//           ...styles.formLabel(theme),
//           whiteSpace: "nowrap",
//           overflow: "hidden",
//           textOverflow: "ellipsis",
//         }}
//       >
//         {label}
//       </div>
//       <div
//         style={{
//           display: "flex",
//           alignItems: "center",
//           gap: 6,
//           flexWrap: "wrap",
//         }}
//       >
//         {value}
//         {stars ?? null}
//         {tooltipHtml ? <RichTooltip html={tooltipHtml} theme={theme} /> : null}
//       </div>
//     </>
//   );
// }

// // components/common/MetricRow.tsx
// import React from "react";
// import type { Theme } from "../../theme/theme";
// import { styles } from "../../theme/styles";
// import { RichTooltip } from "./RichTooltip";

// type MetricRowProps = {
//   theme: Theme;
//   label: string;
//   value: React.ReactNode;
//   tooltipHtml?: string;
//   stars?: React.ReactNode;
// };

// export function MetricRow(props: MetricRowProps): JSX.Element {
//   const { theme, label, value, tooltipHtml, stars } = props;

//   return (
//     <>
//       <div style={styles.formLabel(theme)}>{label}</div>
//       <div
//         style={{
//           display: "flex",
//           alignItems: "center",
//           gap: 6,
//           flexWrap: "wrap",
//         }}
//       >
//         {value}
//         {stars ?? null}
//         {tooltipHtml ? <RichTooltip html={tooltipHtml} theme={theme} /> : null}
//       </div>
//     </>
//   );
// }
