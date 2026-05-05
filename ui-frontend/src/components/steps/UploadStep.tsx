// UploadStep.tsx
import React, { CSSProperties } from "react";
import { SpinnerGlobe } from "../common/SpinnerGlobe";
import { RichTooltip } from "../common/RichTooltip";
import type { Theme } from "../../theme/theme";
import { styles } from "../../theme/styles";

type UiMessage = { text: string; color: string };

type PreviewState = {
  description: string;
  records: Array<Record<string, unknown>>;
};

type UploadStepProps = {
  heading: string;
  buttonText: string;
  buttonTooltipHtml: string;
  disabled: boolean;
  inputRef: React.RefObject<HTMLInputElement>;
  message: UiMessage | null;
  preview: PreviewState | null;
  onUpload: (file: File) => void;
  theme: Theme;

  // Passed from App.tsx to avoid duplication / imports from App
  actionButtonStyle: (theme: Theme, disabled: boolean) => CSSProperties;
  renderMessageLine: (msg: UiMessage | null) => JSX.Element | null;
  renderPreview: (preview: PreviewState | null, theme: Theme) => JSX.Element | null;
};

export function UploadStep(props: UploadStepProps): JSX.Element {
  const isUploadingMessage =
    props.message !== null && props.message.text.startsWith("Uploading");

  return (
    <>
      <h2 style={{ marginTop: 0, marginBottom: 0, ...styles.stepHeading(props.theme) }}>{props.heading}</h2>

      <input
        ref={props.inputRef}
        type="file"
        accept=".csv"
        style={{ display: "none" }}
        onChange={(e) => {
          const input = e.currentTarget;
          const file = input.files?.[0];
          input.value = "";
          if (!file) return;
          props.onUpload(file);
        }}
      />

      <div style={styles.stepActionRow(props.theme)}>
        <button
          type="button"
          disabled={props.disabled}
          onClick={() => {
            if (props.disabled) return;
            props.inputRef.current?.click();
          }}
          style={props.actionButtonStyle(props.theme, props.disabled)}
        >
          {props.buttonText}
        </button>
        <RichTooltip html={props.buttonTooltipHtml} theme={props.theme} />
      </div>

      {isUploadingMessage ? (
        <div
          style={{
            ...styles.spinnerMessageRow(props.theme),
            ...styles.stepMessageGap(props.theme),
            ...(props.message?.color ? { color: props.message.color } : {})
          }}
        >
          <SpinnerGlobe theme={props.theme} />
          <span>{props.message?.text}</span>
        </div>
      ) : (
        props.renderMessageLine(props.message)
      )}

      {props.renderPreview(props.preview, props.theme)}
    </>
  );
}
