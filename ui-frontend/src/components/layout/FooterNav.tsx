import React, { CSSProperties } from "react";
import type { Theme } from "../../theme/theme";

type StepKey = 1 | 2 | 3 | 4;

type FooterNavProps = {
  theme: Theme;
  activeStep: StepKey;
  canBack: boolean;
  canNext: boolean;
  onBack: () => void;
  onNext: () => void;
  navPrimaryButtonStyle: (theme: Theme, disabled: boolean) => CSSProperties;
};

export function FooterNav(props: FooterNavProps): JSX.Element {
  const { theme, activeStep, canBack, canNext, onBack, onNext, navPrimaryButtonStyle } = props;

  return (
    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 48 }}>
      {activeStep !== 1 ? (
        <button
          type="button"
          onClick={() => {
            if (!canBack) return;
            onBack();
          }}
          disabled={!canBack}
          style={navPrimaryButtonStyle(theme, !canBack)}
        >
          ←&nbsp;Back
        </button>
      ) : (
        <div />
      )}

      {activeStep !== 4 ? (
        <button
          type="button"
          onClick={() => {
            if (!canNext) return;
            onNext();
          }}
          disabled={!canNext}
          style={navPrimaryButtonStyle(theme, !canNext)}
        >
          Next&nbsp;→
        </button>
      ) : (
        <div />
      )}
    </div>
  );
}
