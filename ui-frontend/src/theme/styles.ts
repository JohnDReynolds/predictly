// src/theme/styles.ts

import type React from "react";
import type { Theme } from "./theme";
import { rgba } from "./theme";


/**
 * Centralized UI styles.
 * Keep these as "style factories" that accept Theme and return CSSProperties.
 * No component should invent its own fontWeight/color if it matches a token here.
 */
export const styles = {

  // ----- Frames / cards -----
  // Main wizard card frame. Decrease the 0.65 to make the background duller.
  card(theme: Theme): React.CSSProperties {
    return {
      border: `1px solid ${theme.border}`,
      borderRadius: 16,
      padding: 16,
      background: rgba(theme.surface, 0.65)
    };
  },

  // Small secondary/utility button used in tables for "Copy" and for "Debug" button.``
  compactButton(theme: Theme, disabled: boolean): React.CSSProperties {
    return {
      padding: "6px 10px",
      borderRadius: 10,
      border: `1px solid ${theme.border2}`,
      background: disabled ? theme.surface3 : theme.surface2,
      cursor: disabled ? "not-allowed" : "pointer",
      fontSize: 14,
      color: disabled ? theme.text3 : theme.text2,
      fontWeight: 700
    };
  },

  /**
   * Form label
   *
   * Left-hand labels in forms (e.g. "Task:", "Metric:", "Unique ID Column:").
   * This controls typography and a fixed width so fields align.
   */
  formLabel(theme: Theme): React.CSSProperties {
    return {
      fontSize: 16,
      fontWeight: 400,
      width: 180,
      color: theme.text3
    };
  },

  /**
   * Monospace value
   *
   * Used for short data values that should look like code/IDs
   * (e.g. the Target Column name in Step 3).
   */
  monoValue(theme: Theme): React.CSSProperties {
    return {
      fontFamily:
        'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
      fontSize: 14,
      fontWeight: 500,
      color: theme.text
    };
  },

  // Top nav links: Home / Beta / About / Contact
  navLink(theme: Theme, active: boolean): React.CSSProperties {
    return {
      color: active ? theme.linkActive : theme.link,
      textDecoration: active ? "underline" : "none",
      fontWeight: active ? 650 : 500,
      fontSize: 15
    };
  },

  // Back/Next buttons. Compared to primaryButton, the background is darker in dark mode and lighter in light mode.
  navButton(theme: Theme, disabled: boolean): React.CSSProperties {
    return {
      padding: "6px 10px",
      borderRadius: 12,
      border: `1px solid ${theme.border2}`,

      background: disabled
        ? theme.surface3
        : `linear-gradient(
          180deg,
          ${rgba(theme.accent, 0.78)},
          ${rgba(theme.accent2, 0.78)}
        )`,

      cursor: disabled ? "not-allowed" : "pointer",
      color: disabled ? theme.text3 : theme.onAccent,

      fontSize: 12,
      fontWeight: 650,

      // softer + tighter than primary
      boxShadow: disabled ? "none" : `0 6px 14px ${rgba(theme.accent2, 0.14)}`
    };
  },


  /**
   * Panel
   *
   * A small, bordered box used to group related content inside a larger card.
   * Examples include the "Statistics" box in Step 4 and table containers.
   *
   * Panels control only their own frame (border, radius, padding, background),
   * not layout (no margin, width, or grid).
   */
  panel(theme: Theme): React.CSSProperties {
    return {
      border: `1px solid ${theme.border}`,
      borderRadius: 12,
      padding: 12,
      background: theme.surface2
    };
  },

  // Primary big action button (Upload, Predict, etc.).
  primaryButton(theme: Theme, disabled: boolean): React.CSSProperties {
    return {
      padding: "10px 14px",
      borderRadius: 12,
      border: `1px solid ${theme.border2}`,
      background: disabled
        ? theme.surface3
        : `linear-gradient(180deg, ${theme.accent}, ${theme.accent2})`,
      cursor: disabled ? "not-allowed" : "pointer",
      color: disabled ? theme.text3 : theme.onAccent,
      fontSize: 14,
      fontWeight: 650,
      boxShadow: disabled ? "none" : `0 10px 26px ${rgba(theme.accent2, 0.18)}`
    };
  },

  // Row style for "Uploading..." / "Predicting..." spinner messages.
  spinnerMessageRow(theme: Theme): React.CSSProperties {
    return {
      marginTop: 10,
      display: "flex",
      alignItems: "center",
      gap: 10,

      // formerly styles.label(theme)
      fontSize: 14,
      fontWeight: 500,
      color: theme.text2
    };
  },

  // Layout row for the primary step actions (button row directly under each step title).
  stepActionRow(theme: Theme): React.CSSProperties {
    return {
      marginTop: 12, // vertical spacing between numbered "Step Title" and button directly below it.
      display: "flex",
      alignItems: "center",
      gap: 6, // the horizontal space between the button and the tooltip
      flexWrap: "wrap"
    };
  },

  // Step headings: "1. Training File", "2. Prediction File", etc.
  stepHeading(theme: Theme): React.CSSProperties {
    return {
      fontSize: 22,
      fontWeight: 550,
      color: theme.text
    };
  },

  // Vertical spacing between the primary action button and status / messages.
  stepMessageGap(theme: Theme): React.CSSProperties {
    return {
      marginTop: 12
    };
  },

  // Vertical spacing between major sections/tables within a step.
  stepSectionGap(theme: Theme): React.CSSProperties {
    return {
      marginTop: 40,
      marginBottom: 40
    };
  },

  // Subtle section titles, e.g. "Statistics" and sortableTable section titles.
  subtleTitle(theme: Theme): React.CSSProperties {
    return {
      fontSize: 19,
      fontWeight: 400,
      color: theme.text,
      marginBottom: 20,
      textAlign: "center"
    };
  },

} as const;
