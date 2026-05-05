// PredictStep.tsx
import React, { CSSProperties } from "react";
import { RichTooltip } from "../common/RichTooltip";
import { SpinnerGlobe } from "../common/SpinnerGlobe";
import { styles } from "../../theme/styles";

type UiMessage = { text: string; color: string };

type TrainStatusState = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "BUSY" | "UNKNOWN";

type TrainStatusPayload = {
  status?: string;
  state: TrainStatusState;
  updated_at_epoch?: number;
  error_type?: string;
  message?: string;
  [key: string]: unknown;
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

type PredictStepProps = {
  theme: Theme;
  tooltips: Record<string, string>;

  // State / flags
  isTraining: boolean;
  isDebug: boolean;
  canTrain: boolean;
  canInlineBack: boolean;
  showInlineBack: boolean;

  // Messages / status
  trainMsg: UiMessage | null;
  trainStatus: TrainStatusPayload | null;
  trainStatusMsg: UiMessage | null;

  // Callbacks
  onPredict: () => void;
  onBackToParams: () => void;

  // Helpers from App.tsx
  actionButtonStyle: (theme: Theme, disabled: boolean) => CSSProperties;
  backButtonStyle: (theme: Theme, disabled: boolean) => CSSProperties;
  renderMessageLine: (msg: UiMessage | null) => JSX.Element | null;
  renderOutput: () => JSX.Element | null;

};

export function PredictStep(props: PredictStepProps): JSX.Element {
  const {
    theme,
    tooltips,
    isTraining,
    isDebug,
    canTrain,
    canInlineBack,
    showInlineBack,
    trainMsg,
    trainStatus,
    trainStatusMsg,
    onPredict,
    onBackToParams,
    renderMessageLine,
    renderOutput,
    actionButtonStyle,
    backButtonStyle
  } = props;

  const showNonDebugBusyLine = !isDebug && isTraining;
  const showNonDebugFinalMsg = !isDebug && !!trainMsg && !isTraining;

  return (
    <>
      <h2 style={{ marginTop: 0, marginBottom: 0, ...styles.stepHeading(theme) }}>4. Predict</h2>

      <div style={{ ...styles.stepActionRow(theme), justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={() => {
              if (!canTrain) return;
              onPredict();
            }}
            disabled={!canTrain}
            style={actionButtonStyle(theme, !canTrain)}
          >
            Predict
          </button>
          <RichTooltip html={tooltips.step4Button} theme={theme} />
        </div>

        {showInlineBack ? (
          <button
            type="button"
            onClick={() => {
              if (!canInlineBack) return;
              onBackToParams();
            }}
            disabled={!canInlineBack}
            style={backButtonStyle(theme, !canInlineBack)}
            title="Back to Parameters"
          >
            ←&nbsp;Back
          </button>
        ) : null}
      </div>

      {isDebug ? (
        <>
          {renderMessageLine(trainMsg)}
          {trainStatusMsg ? renderMessageLine(trainStatusMsg) : null}
          {trainStatus ? (
            <div
              style={{
                ...styles.stepMessageGap(theme),
                fontSize: 14,
                color: theme.text2
              }}
            >
              Status: <b style={{ color: theme.text }}>{trainStatus.state}</b>
              {typeof trainStatus.updated_at_epoch === "number" ? (
                <span style={{ color: theme.text3 }}> (token={trainStatus.updated_at_epoch})</span>
              ) : null}
            </div>
          ) : null}
        </>
      ) : (
        <>
          {showNonDebugBusyLine ? (
            <div
              style={{
                ...styles.spinnerMessageRow(theme),
                ...styles.stepMessageGap(theme)
              }}
            >
              <SpinnerGlobe theme={theme} />
              <span>Predicting... Please be patient, this may take several minutes...</span>
            </div>
          ) : null}
          {showNonDebugFinalMsg ? renderMessageLine(trainMsg) : null}
        </>
      )}

      {renderOutput()}
    </>
  );
}
