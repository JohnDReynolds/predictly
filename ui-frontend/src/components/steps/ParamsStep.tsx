// // ParamsStep.tsx
// import React, { CSSProperties } from "react";
// import { RichTooltip } from "../common/RichTooltip";
// import { styles } from "../../theme/styles";

// type UiMessage = { text: string; color: string };

// type Step2Meta = {
//   validTaskMetrics: Record<string, string[]>;
//   uniqueColumns: string[];
//   yColumnName: string;
// };

// type ParamsState = {
//   task: string;
//   metric: string;
//   uidColumnName: string;
// };

// type Theme = {
//   bg: string;
//   surface: string;
//   surface2: string;
//   surface3: string;
//   border: string;
//   border2: string;
//   text: string;
//   text2: string;
//   text3: string;
//   link: string;
//   linkActive: string;
//   accent: string;
//   accent2: string;
//   danger: string;
//   onAccent: string;
// };

// type Tooltips = Record<string, string>;

// type ParamsStepProps = {
//   theme: Theme;
//   tooltips: Tooltips;
//   anyBusy: boolean;
//   step2Meta: Step2Meta | null;
//   params: ParamsState | null;
//   paramsMsg: UiMessage | null;
//   onTaskChange: (task: string) => void;
//   onMetricChange: (metric: string) => void;
//   onUidColumnChange: (uidColumnName: string) => void;
//   renderMessageLine: (msg: UiMessage | null) => JSX.Element | null;
// };

// export function ParamsStep(props: ParamsStepProps): JSX.Element {
//   const { theme, tooltips, anyBusy, step2Meta, params, paramsMsg } = props;

//   if (!step2Meta || !params) {
//     return (
//       <>
//         <h2 style={{ marginTop: 0, ...styles.stepHeading(theme) }}>3. Parameters</h2>
//         <div style={{ color: theme.text3, fontSize: 15 }}>
//           Upload a Prediction File (Step 2) to infer valid tasks, metrics, and columns.
//         </div>
//         {props.renderMessageLine(paramsMsg)}
//       </>
//     );
//   }

//   const rowStyle: CSSProperties = {
//     display: "flex",
//     alignItems: "center",
//     gap: 12,
//     flexWrap: "wrap"
//   };

//   // Option A: enlarge the closed control consistently across browsers.
//   // Note: the *opened* dropdown list may still be OS-rendered (especially in Safari/macOS),
//   // so it can ignore font styling. We still apply option font size for browsers that respect it.
//   const pickerFontSizePx = 14;

//   const controlStyle: CSSProperties = {
//     height: 22, // height of picker box when closed
//     padding: "0 12px", // horizontal only; vertical is controlled by height
//     boxSizing: "border-box",

//     borderRadius: 12,
//     border: `1px solid ${theme.border}`,
//     minWidth: 180,
//     background: theme.surface2,
//     color: theme.text,
//     outline: "none",

//     fontSize: pickerFontSizePx
//     // no lineHeight needed; height controls the box
//   };

//   const optionStyle: CSSProperties = {
//     // Some browsers apply this to the opened list; Safari/macOS may ignore it.
//     fontSize: pickerFontSizePx
//   };

//   const taskKeys = Object.keys(step2Meta.validTaskMetrics);
//   const metricsForSelectedTask =
//     params.task && step2Meta.validTaskMetrics[params.task]
//       ? step2Meta.validTaskMetrics[params.task]
//       : [];

//   const isTaskPickerDisabled = anyBusy || taskKeys.length <= 1;

//   return (
//     <>
//       <h2 style={{ marginTop: 0, ...styles.stepHeading(theme) }}>3. Parameters</h2>

//       <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12, maxWidth: 450 }}>
//         {/* Target Column */}
//         <div style={rowStyle}>
//           <div style={styles.formLabel(theme)}>Target Column:</div>
//           <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
//             <div style={styles.monoValue(theme)}>{step2Meta.yColumnName}</div>
//             <RichTooltip html={tooltips.yColumnName} theme={theme} />
//           </div>
//         </div>

//         {/* Task Picker */}
//         <div style={rowStyle}>
//           <div style={styles.formLabel(theme)}>Model Task:</div>
//           <select
//             value={params.task}
//             onChange={(e) => {
//               props.onTaskChange(e.currentTarget.value);
//             }}
//             style={controlStyle}
//             disabled={isTaskPickerDisabled}
//           >
//             {taskKeys.map((t) => (
//               <option key={t} value={t} style={optionStyle}>
//                 {t}
//               </option>
//             ))}
//           </select>
//           <RichTooltip html={tooltips.task} theme={theme} />
//         </div>

//         {/* Metric Picker (depends on Task) */}
//         <div style={rowStyle}>
//           <div style={styles.formLabel(theme)}>Metric:</div>
//           <select
//             value={params.metric}
//             onChange={(e) => {
//               props.onMetricChange(e.currentTarget.value);
//             }}
//             style={controlStyle}
//             disabled={anyBusy}
//           >
//             {metricsForSelectedTask.map((m) => (
//               <option key={m} value={m} style={optionStyle}>
//                 {m}
//               </option>
//             ))}
//           </select>
//           <RichTooltip html={tooltips.metric} theme={theme} />
//         </div>

//         {/* Unique ID Column */}
//         <div style={rowStyle}>
//           <div style={styles.formLabel(theme)}>Unique ID Column:</div>
//           <select
//             value={params.uidColumnName}
//             onChange={(e) => {
//               props.onUidColumnChange(e.currentTarget.value);
//             }}
//             style={controlStyle}
//             disabled={anyBusy}
//           >
//             <option value="" style={optionStyle}>
//               (none)
//             </option>
//             {step2Meta.uniqueColumns.map((c) => (
//               <option key={c} value={c} style={optionStyle}>
//                 {c}
//               </option>
//             ))}
//           </select>
//           <RichTooltip html={tooltips.uid} theme={theme} />
//         </div>
//       </div>

//       {props.renderMessageLine(paramsMsg)}
//     </>
//   );
// }

// ParamsStep.tsx
import React, { CSSProperties } from "react";
import { RichTooltip } from "../common/RichTooltip";
import { styles } from "../../theme/styles";

type UiMessage = { text: string; color: string };

type Step2Meta = {
  validTaskMetrics: Record<string, string[]>;
  uniqueColumns: string[];
  yColumnName: string;
};

type ParamsState = {
  task: string;
  metric: string;
  uidColumnName: string;
};

type Theme = {
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
  onAccent: string;
};

type Tooltips = Record<string, string>;

type ParamsStepProps = {
  theme: Theme;
  tooltips: Tooltips;
  anyBusy: boolean;
  step2Meta: Step2Meta | null;
  params: ParamsState | null;
  paramsMsg: UiMessage | null;
  onTaskChange: (task: string) => void;
  onMetricChange: (metric: string) => void;
  onUidColumnChange: (uidColumnName: string) => void;
  renderMessageLine: (msg: UiMessage | null) => JSX.Element | null;
};

export function ParamsStep(props: ParamsStepProps): JSX.Element {
  const { theme, tooltips, anyBusy, step2Meta, params, paramsMsg } = props;

  if (!step2Meta || !params) {
    return (
      <>
        <h2 style={{ marginTop: 0, ...styles.stepHeading(theme) }}>3. Parameters</h2>
        <div style={{ color: theme.text3, fontSize: 15 }}>
          Upload a Prediction File (Step 2) to infer valid tasks, metrics, and columns.
        </div>
        {props.renderMessageLine(paramsMsg)}
      </>
    );
  }

  const rowStyle: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap"
  };

  // Option A: enlarge the closed control consistently across browsers.
  // Note: the *opened* dropdown list may still be OS-rendered (especially in Safari/macOS),
  // so it can ignore font styling. We still apply option font size for browsers that respect it.
  const pickerFontSizePx = 14;

  const controlStyle: CSSProperties = {
    height: 22, // height of picker box when closed
    padding: "0 12px", // horizontal only; vertical is controlled by height
    boxSizing: "border-box",

    borderRadius: 12,
    border: `1px solid ${theme.border}`,
    minWidth: 180,
    background: theme.surface2,
    color: theme.text,
    outline: "none",

    fontSize: pickerFontSizePx
    // no lineHeight needed; height controls the box
  };

  const optionStyle: CSSProperties = {
    // Some browsers apply this to the opened list; Safari/macOS may ignore it.
    fontSize: pickerFontSizePx
  };

  const taskKeys = Object.keys(step2Meta.validTaskMetrics);
  const hasSingleTask = taskKeys.length === 1;
  const displayedTask = params.task || taskKeys[0] || "";

  const metricsForSelectedTask =
    displayedTask && step2Meta.validTaskMetrics[displayedTask]
      ? step2Meta.validTaskMetrics[displayedTask]
      : [];

  return (
    <>
      <h2 style={{ marginTop: 0, ...styles.stepHeading(theme) }}>3. Parameters</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12, maxWidth: 450 }}>
        {/* Target Column */}
        <div style={rowStyle}>
          <div style={styles.formLabel(theme)}>Target Column:</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <div style={styles.monoValue(theme)}>{step2Meta.yColumnName}</div>
            <RichTooltip html={tooltips.yColumnName} theme={theme} />
          </div>
        </div>

        {/* Task Picker / Value */}
        <div style={rowStyle}>
          <div style={styles.formLabel(theme)}>Model Task:</div>
          {hasSingleTask ? (
            <div style={styles.monoValue(theme)}>{displayedTask}</div>
          ) : (
            <select
              value={params.task}
              onChange={(e) => {
                props.onTaskChange(e.currentTarget.value);
              }}
              style={controlStyle}
              disabled={anyBusy}
            >
              {taskKeys.map((t) => (
                <option key={t} value={t} style={optionStyle}>
                  {t}
                </option>
              ))}
            </select>
          )}
          <RichTooltip html={tooltips.task} theme={theme} />
        </div>

        {/* Metric Picker (depends on Task) */}
        <div style={rowStyle}>
          <div style={styles.formLabel(theme)}>Metric:</div>
          <select
            value={params.metric}
            onChange={(e) => {
              props.onMetricChange(e.currentTarget.value);
            }}
            style={controlStyle}
            disabled={anyBusy}
          >
            {metricsForSelectedTask.map((m) => (
              <option key={m} value={m} style={optionStyle}>
                {m}
              </option>
            ))}
          </select>
          <RichTooltip html={tooltips.metric} theme={theme} />
        </div>

        {/* Unique ID Column */}
        <div style={rowStyle}>
          <div style={styles.formLabel(theme)}>Unique ID Column:</div>
          <select
            value={params.uidColumnName}
            onChange={(e) => {
              props.onUidColumnChange(e.currentTarget.value);
            }}
            style={controlStyle}
            disabled={anyBusy}
          >
            <option value="" style={optionStyle}>
              (none)
            </option>
            {step2Meta.uniqueColumns.map((c) => (
              <option key={c} value={c} style={optionStyle}>
                {c}
              </option>
            ))}
          </select>
          <RichTooltip html={tooltips.uid} theme={theme} />
        </div>
      </div>

      {props.renderMessageLine(paramsMsg)}
    </>
  );
}
