/**  RichTooltip.tsx  **/
import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type Theme = {
  bg: string;
  surface: string;
  surface2: string;
  border2: string;
  text2: string;
  text3: string;
  accent: string;
};

function isDarkBg(hex: string): boolean {
  const h = hex.replace("#", "");
  if (h.length !== 6) return false;
  const r = Number.parseInt(h.slice(0, 2), 16);
  const g = Number.parseInt(h.slice(2, 4), 16);
  const b = Number.parseInt(h.slice(4, 6), 16);
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
  return luminance < 90; // threshold; tweak if needed
}

function rgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const r = Number.parseInt(h.slice(0, 2), 16);
  const g = Number.parseInt(h.slice(2, 4), 16);
  const b = Number.parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

type RichTooltipProps = {
  html: string;
  theme: Theme;
};

export function RichTooltip(props: RichTooltipProps): JSX.Element {
  const { html, theme } = props;

  const [open, setOpen] = useState<boolean>(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(
    null
  );

  const rootRef = useRef<HTMLSpanElement | null>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;

    const onDocPointerDown = (e: PointerEvent): void => {
      const root = rootRef.current;
      if (!root) {
        setOpen(false);
        return;
      }
      const target = e.target as Node | null;
      if (target && root.contains(target)) return;
      setOpen(false);
    };

    // Attach in the bubble phase (no `true` as the third argument)
    document.addEventListener("pointerdown", onDocPointerDown);
    return () => {
      document.removeEventListener("pointerdown", onDocPointerDown);
    };
  }, [open]);

  const isDark = isDarkBg(theme.bg);

  // Dark mode: help control easier to see
  const helpBg = isDark ? rgba("#ffffff", 0.12) : theme.surface2;
  const helpBorder = isDark ? rgba("#ffffff", 0.22) : theme.border2;
  const helpText = isDark ? rgba("#ffffff", 0.82) : theme.accent;

  function toggleOpen(event: React.MouseEvent<HTMLButtonElement>): void {
    event.preventDefault();
    event.stopPropagation();

    setOpen((prev) => {
      const next = !prev;

      if (next) {
        const root = rootRef.current;
        const anchorEl = root ?? (event.currentTarget as HTMLElement);
        const rect = anchorEl.getBoundingClientRect();

        // Base position: below the button
        let left = rect.left;
        const top = rect.bottom + 6;

        const TOOLTIP_WIDTH = 390;
        const PADDING = 8;
        const maxLeft = window.innerWidth - TOOLTIP_WIDTH - PADDING;

        if (left > maxLeft) {
          left = Math.max(PADDING, maxLeft);
        }
        if (left < PADDING) {
          left = PADDING;
        }

        setCoords({ top, left });
      }

      return next;
    });
  }

  // The anchor (question mark button) lives inline where the component is used.
  // The tooltip popup itself is rendered into document.body via a portal.
  return (
    <span
      ref={rootRef}
      style={{
        display: "inline-block",
        marginLeft: 8,
        position: "relative",
      }}
    >
      <button
        type="button"
        aria-label="Help"
        onClick={toggleOpen}
        style={{
          cursor: "pointer",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 18,
          height: 18,
          borderRadius: 9,
          border: `1px solid ${helpBorder}`,
          color: helpText,
          fontWeight: 900,
          fontSize: 12,
          background: helpBg,
          userSelect: "none",
          padding: 0,
        }}
      >
        ?
      </button>

      {open && coords && typeof document !== "undefined"
        ? createPortal(
          <div
            role="dialog"
            aria-label="Tooltip"
            style={{
              position: "fixed", // not affected by any parent overflow
              zIndex: 2000,
              top: coords.top,
              left: coords.left,
              width: 390,
              maxWidth: "78vw",
              border: `1px solid ${theme.border2}`,
              borderRadius: 12,
              padding: 12,
              background: theme.surface,
              boxShadow: `0 14px 42px ${rgba("#000000", 0.55)}`,
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <div
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: html }}
              style={{
                fontSize: 14,
                fontWeight: 400,         // normalize across all tooltips
                color: theme.text2,
                lineHeight: 1.25,
                textAlign: "left",
                whiteSpace: "normal",    // wrap lines
                wordBreak: "break-word", // prevent long tokens from overflowing
              }}
            />
          </div>,
          document.body
        )
        : null}
    </span>
  );
}
