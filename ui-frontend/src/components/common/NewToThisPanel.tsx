/**  NewToThisPanel.tsx
 * A collapsible panel that explains tabular machine learning to new users.
 * Used in the first step of the wizard (upload training file).
 **/
import React from "react";
import { styles } from "../../theme/styles";
import { type Theme } from "../../theme/theme";

export function NewToThisPanel({ theme }: { theme: Theme }): JSX.Element {
  const [open, setOpen] = React.useState<boolean>(false);

  return (
    <div style={{ marginBottom: 56 }}>
      {/* Collapsed header */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          padding: "10px 12px",
          borderRadius: 10,
          background: theme.surface2,
          border: `1px solid ${theme.border}`,
          fontWeight: 700,
          color: theme.text
        }}
      >
        <span>New to this?</span>
        <span style={{ fontSize: 14, color: theme.text2 }}>
          {open ? "▲" : "▼"}
        </span>
      </div>

      {/* Expanded content */}
      {
        open ? (
          <div
            style={{
              ...styles.panel(theme),
              marginTop: 2,
              padding: 14,
              fontSize: 14,
              color: theme.text2,
              lineHeight: 1.5
            }}
          >
            <p>
              Predictly uses <b>tabular machine learning</b> to predict an outcome.  It uses spreadsheet-style data,
              where each <b>row</b> is a sample (e.g. a house, a customer, or a support ticket), and each <b>column</b> is a feature.
              One special column is the <b>target</b>, which is the outcome that Predictly will predict.
            </p>

            <p>
              <b>Training data</b> has features and the target.  Predictly will study the training features to
              learn how they can be used to predict the target value.
            </p>

            <p>
              <b>Prediction data</b> has the same features, but without the target.  Prediclty uses what it has
              learned from the training features to predict this missing target.
            </p>

            <hr
              style={{
                border: "none",
                borderTop: `1px solid ${theme.border}`,
                margin: "12px 0"
              }}
            />

            <p style={{ marginBottom: 2 }}>
              <b>Regression Example (numeric target values)</b>
            </p>

            <pre
              style={{
                marginTop: 0,
                marginBottom: 10,
                fontSize: 12,
                background: theme.surface3,
                padding: 8,
                borderRadius: 8,
                whiteSpace: "pre"
              }}
            >
              {[
                "id  bedrooms  sqft  quality   price (target)",
                "C1  3         1600  high      350000",
                "C2  2         900   medium    275000",
                "C3  4         1850  low       305000"
              ].join("\n")}
            </pre>

            <p style={{ marginBottom: 2 }}>
              <b>Binary Classification Example (two possible target values)</b>
            </p>
            <pre
              style={{
                marginTop: 0,
                marginBottom: 10,
                fontSize: 12,
                background: theme.surface3,
                padding: 8,
                borderRadius: 8,
                whiteSpace: "pre"
              }}
            >
              {[
                "customer_id  tenure  plan   churned (target)",
                "A12          18      pro    No",
                "B07          3       basic  Yes",
                "B43          9       pro    No"
              ].join("\n")}
            </pre>

            <p style={{ marginBottom: 2 }}>
              <b>Multi-Class Classification Example (multiple possible target values)</b>
            </p>
            <pre
              style={{
                marginTop: 0,
                marginBottom: 10,
                fontSize: 12,
                background: theme.surface3,
                padding: 8,
                borderRadius: 8,
                whiteSpace: "pre"
              }}
            >
              {[
                "ticket_id  channel  plan   priority (target)",
                "T101       email    pro    Medium",
                "T102       phone    basic  Low",
                "T103       phone    pro    High"
              ].join("\n")}
            </pre>

            <div style={{ marginTop: 10, fontStyle: "italic", color: theme.text3 }}>
              Collapse this panel and start by uploading your Training File below.
            </div>
          </div>
        ) : null
      }
    </div >
  );
}
