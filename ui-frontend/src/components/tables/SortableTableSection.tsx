// SortableTableSection.tsx
import React, { useMemo, useState } from "react";
import { styles } from "../../theme/styles";
import { rgba, type Theme } from "../../theme/theme";
import { RichTooltip } from "../common/RichTooltip";

type SortDir = "asc" | "desc";

function escapeCsvCell(raw: string): string {
  const s = raw.replace(/"/g, '""');
  const mustQuote =
    s.includes(",") || s.includes("\n") || s.includes("\r") || s.includes('"');
  return mustQuote ? `"${s}"` : s;
}

function recordsToCsv(
  records: Array<Record<string, unknown>>,
  columns: string[]
): string {
  const header = columns.map((c) => escapeCsvCell(c)).join(",");
  const lines = records.map((r) =>
    columns
      .map((c) => {
        const v = r[c];
        return escapeCsvCell(String(v ?? ""));
      })
      .join(",")
  );
  return [header, ...lines].join("\n");
}

async function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && globalThis.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  ta.style.top = "-9999px";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

function asComparableNumber(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x !== "string") return null;
  const s = x.trim();
  if (!s) return null;
  const n = Number(s);
  if (Number.isFinite(n)) return n;
  return null;
}

type DecimalAlignedNumberProps = {
  value: number;
  decimals: number;
};

/**
 * Simpler, more robust numeric renderer.
 *
 * - Uses a single string with fixed decimals.
 * - Relies on right-aligned cell + tabular-nums for visual alignment.
 * - Avoids inline-grid, which can misbehave with overflow/ellipsis in some browsers.
 */
function DecimalAlignedNumber({
  value,
  decimals
}: DecimalAlignedNumberProps): JSX.Element {
  if (!Number.isFinite(value)) {
    return <span>—</span>;
  }

  const safeDecimals = Math.max(0, Math.floor(decimals));

  const formatted = value.toLocaleString("en-US", {
    minimumFractionDigits: safeDecimals,
    maximumFractionDigits: safeDecimals
  });

  return (
    <span
      style={{
        display: "inline-block",
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap"
      }}
    >
      {formatted}
    </span>
  );
}

/**
 * Shared utility: collect ordered column names from record array.
 * Exported so App.tsx can reuse it for preview tables.
 */
export function getColumns(
  records: Array<Record<string, unknown>>
): string[] {
  const seen = new Set<string>();
  const cols: string[] = [];
  for (const r of records) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) {
        seen.add(k);
        cols.push(k);
      }
    }
  }
  return cols;
}

function compareValuesNumericFirst(a: unknown, b: unknown): number {
  const an = asComparableNumber(a);
  const bn = asComparableNumber(b);

  if (an !== null && bn !== null) return an - bn;
  if (an !== null && bn === null) return -1;
  if (an === null && bn !== null) return 1;

  const as = String(a ?? "").toLowerCase();
  const bs = String(b ?? "").toLowerCase();
  if (as < bs) return -1;
  if (as > bs) return 1;
  return 0;
}

function sortRecords(
  records: Array<Record<string, unknown>>,
  col: string,
  dir: SortDir
): Array<Record<string, unknown>> {
  const decorated = records.map((r, idx) => ({ r, idx }));
  decorated.sort((x, y) => {
    const c = compareValuesNumericFirst(x.r[col], y.r[col]);
    if (c !== 0) return dir === "asc" ? c : -c;
    return x.idx - y.idx;
  });
  return decorated.map((d) => d.r);
}

type SortableTableSectionProps = {
  title: string;
  records: Array<Record<string, unknown>>;
  emptyText: string;
  theme: Theme;
  /** Controls visibility of Copy CSV link. */
  showCopyButton?: boolean;
  /** Render in "ghost" sample mode. */
  isExample?: boolean;
  /**
   * When true, do NOT apply a default sort column on initial render.
   * Rows will appear in the order provided until the user clicks a header.
   * Use this for Step 1/2 previews so CSV row order is preserved initially.
   */
  disableInitialSort?: boolean;
  /**
   * Names of columns that should be displayed as percentages (0.1234 -> 12.34%).
   * Values are assumed to be in [0, 1] or numeric strings.
   */
  percentageColumns?: string[];
  /**
   * Tooltip map.
   *
   * Recommended keys:
   * - "tableOverview"          → tooltip next to the title
   * - "column:<column_name>"   → tooltip next to a column header
   *
   * Example:
   * {
   *   tableOverview: "...",
   *   "column:Importance": "...",
   *   "column:Prediction": "..."
   * }
   */
  tooltips?: Record<string, string>;
  /**
   * Optional per-column max widths (in px) for cells.
   * Example: { Comment: 420, "Very Long Text Column": 520 }
   */
  columnMaxWidths?: Record<string, number>;
  /**
   * When true, all *fractional* numeric columns use the same number
   * of decimal places (the maximum inferred for any fractional column).
   * Integer-only numeric columns stay at 0 decimals.
   */
  lockNumericDecimalsToGlobalMax?: boolean;
};

/**
 * Sortable table with CSV export and sticky header.
 * Used for Predictions, Feature Affects, and previews.
 */
export function SortableTableSection(
  props: SortableTableSectionProps
): JSX.Element {
  const {
    title,
    records,
    emptyText,
    theme,
    showCopyButton = true,
    isExample = false,
    disableInitialSort = false,
    percentageColumns = [],
    tooltips = {},
    columnMaxWidths = {},
    lockNumericDecimalsToGlobalMax = false
  } = props;

  const cols = useMemo(() => getColumns(records), [records]);

  // Column-level numeric detection:
  // A column is "numeric" only if every non-null value is numeric.
  const columnIsNumeric = useMemo(() => {
    const map: Record<string, boolean> = {};

    for (const c of cols) {
      map[c] = records.every((r) => {
        const v = r[c];
        return v == null || asComparableNumber(v) !== null;
      });
    }

    return map;
  }, [cols, records]);

  // Per-column decimal precision decision:
  // - If all numeric values are integers -> 0 decimals
  // - Otherwise:
  //   - Base rule:
  //       - max abs < 1 -> 3 decimals
  //       - else -> 2 decimals
  //   - If smallest non-zero abs < 0.001, bump decimals until visible (cap at 6)
  const columnDecimalPlaces = useMemo(() => {
    const map: Record<string, number> = {};

    for (const c of cols) {
      let maxAbs = 0;
      let minNonZeroAbs = Number.POSITIVE_INFINITY;
      let hasNumeric = false;
      let hasFractional = false;

      for (const r of records) {
        const n = asComparableNumber(r[c]);
        if (n !== null) {
          hasNumeric = true;

          const abs = Math.abs(n);
          if (abs > maxAbs) {
            maxAbs = abs;
          }
          if (abs > 0 && abs < minNonZeroAbs) {
            minNonZeroAbs = abs;
          }

          // Detect fractional values
          if (!Number.isInteger(n)) {
            hasFractional = true;
          }
        }
      }

      if (!hasNumeric) {
        map[c] = 0;
        continue;
      }

      // If everything is an integer, show no decimals
      if (!hasFractional) {
        map[c] = 0;
        continue;
      }

      // Base decimals
      let decimals = maxAbs < 1 ? 3 : 2;

      // Bump precision for very small values
      if (minNonZeroAbs < 0.001) {
        let v = minNonZeroAbs;
        while (v < 0.001 && decimals < 6) {
          v *= 10;
          decimals += 1;
        }
      }

      map[c] = decimals;
    }

    return map;
  }, [cols, records]);

  // Global max decimals among numeric columns that actually
  // need fractional digits (i.e., > 0). Integer-only columns
  // have 0 and are ignored here.
  const globalMaxDecimalPlaces = useMemo(() => {
    let max = 0;
    for (const c of cols) {
      const d = columnDecimalPlaces[c];
      if (typeof d === "number" && d > max) {
        max = d;
      }
    }
    return max;
  }, [cols, columnDecimalPlaces]);

  const [sortCol, setSortCol] = useState<string>("");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  useEffectForColumns(
    cols,
    sortCol,
    setSortCol,
    setSortDir,
    disableInitialSort
  );

  const sorted = useMemo(() => {
    if (!records.length) return records;
    if (!sortCol) return records; // no active sort -> preserve original row order
    return sortRecords(records, sortCol, sortDir);
  }, [records, sortCol, sortDir]);

  const canCopy = cols.length > 0 && sorted.length > 0;
  const showCopyLink = showCopyButton && canCopy && !isExample;

  function onToggleSort(col: string): void {
    if (!col) return;

    if (sortCol !== col) {
      setSortCol(col);
      setSortDir(col === "Model Importance" ? "desc" : "asc");
      return;
    }
    setSortDir((d) => (d === "asc" ? "desc" : "asc"));
  }

  function sortIndicator(col: string): string {
    if (col !== sortCol) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  }

  function handleCopy(): void {
    if (!canCopy) return;
    const csv = recordsToCsv(sorted, cols);

    void copyTextToClipboard(csv)
      .then(() => alert(`${title} copied to clipboard as CSV.`))
      .catch(() => alert("Could not copy to clipboard."));
  }

  function isPercentageColumn(col: string): boolean {
    return percentageColumns.includes(col);
  }

  return (
    // Left-justified in the page (normal block flow)
    <div>
      {/* Shrink-wrapped container for the action row + card */}
      <div
        style={{
          display: "inline-block",
          maxWidth: "100%"
        }}
      >
        {/* Copy link ABOVE the card, aligned with card/table right edge */}
        {showCopyLink ? (
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginBottom: 4 // vertical gap between link bottom and card top
            }}
          >
            <button
              type="button"
              onClick={handleCopy}
              style={{
                border: "none",
                background: "none",
                padding: 0,
                margin: 0,
                cursor: "pointer",
                fontSize: 11,
                color: theme.text3,
                textDecoration: "underline",
                lineHeight: 1
              }}
              title="Copy as CSV"
            >
              Copy
            </button>
          </div>
        ) : null}

        {/* Relative wrapper so the sample watermark overlays the card only */}
        <div
          style={{
            position: "relative"
          }}
        >
          {isExample ? (
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                pointerEvents: "none",
                zIndex: 1,
                fontSize: 32,
                fontWeight: 800,
                textTransform: "uppercase",
                letterSpacing: 2,
                color: rgba(theme.text3, 0.3),
                transform: "rotate(-20deg)",
                userSelect: "none",
                textAlign: "center",
                padding: 16,
                lineHeight: 1.2,
                fontStyle: "italic" // italic watermark for example data
              }}
            >
              Sample Data Only
            </div>
          ) : null}
          <div
            style={{
              ...styles.panel(theme),
              padding: 8,
              maxHeight: 340, // height-limited card
              // Shrink-wrap to the table, but don't exceed viewport width
              width: "fit-content",
              maxWidth: "100%",
              // Scrollbars when content exceeds panel
              overflowX: "auto",
              overflowY: "auto",
              // Ghost effect for sample mode
              opacity: isExample ? 0.55 : 1,
              filter: isExample ? "grayscale(0.15)" : "none",
              position: "relative"
            }}
          >
            {/* Title centered over the card/table, with optional tooltip */}
            <div
              style={{
                ...styles.subtleTitle(theme),
                color: isExample
                  ? theme.text3
                  : styles.subtleTitle(theme).color,
                fontStyle: isExample ? "italic" : "normal",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                gap: 4,
                whiteSpace: "nowrap" // keep text on one line
                // IMPORTANT: no overflow / textOverflow here,
                // or we will clip the tooltip popup.
              }}
            >
              <span
                style={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  maxWidth: "100%"
                }}
              >
                {title}
              </span>
              {tooltips.tableOverview ? (
                <span style={{ whiteSpace: "normal" }}>
                  <RichTooltip
                    html={tooltips.tableOverview}
                    theme={theme}
                  />
                </span>
              ) : null}
            </div>

            {!records.length ? (
              <div
                style={{
                  padding: 8,
                  color: theme.text3,
                  fontStyle: "italic",
                  fontSize: 14
                }}
              >
                {emptyText}
              </div>
            ) : (
              <div>
                <table
                  style={{
                    borderCollapse: "collapse",
                    width: "max-content",
                    fontSize: 14,
                    color: theme.text2,
                    tableLayout: "auto"
                  }}
                >
                  <thead>
                    <tr>
                      {cols.map((c) => {
                        const columnTooltipKey = `column:${c}`;
                        return (
                          <th
                            key={c}
                            role="button"
                            tabIndex={0}
                            onClick={() => onToggleSort(c)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                onToggleSort(c);
                              }
                            }}
                            style={{
                              textAlign: "left",
                              borderBottom: `1px solid ${theme.border}`,
                              padding: "8px 8px",
                              position: "sticky",
                              top: 0,
                              background: theme.surface3,
                              cursor: "pointer",
                              userSelect: "none",
                              whiteSpace: "nowrap",
                              fontSize: 13,
                              color: theme.text2
                            }}
                            title="Click to sort"
                          >
                            <div
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 4
                              }}
                            >
                              <span>
                                {c}
                                {sortIndicator(c)}
                              </span>
                              {tooltips[columnTooltipKey] ? (
                                <span style={{ whiteSpace: "normal" }}>
                                  <RichTooltip
                                    html={tooltips[columnTooltipKey]}
                                    theme={theme}
                                  />
                                </span>
                              ) : null}
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((r, idx) => (
                      <tr key={idx}>
                        {cols.map((c) => {
                          const isPct = isPercentageColumn(c);
                          const rawValue = r[c];
                          const numericValue = asComparableNumber(rawValue);
                          const isNumericValue = numericValue !== null;
                          const isNumericColumn = columnIsNumeric[c];

                          let content: React.ReactNode;
                          let titleText = "";

                          if (isPct && isNumericValue) {
                            // Display as percentage with 2 decimals, e.g. 12.34%
                            const pctValue = numericValue * 100;
                            const pctText = pctValue.toLocaleString("en-US", {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2
                            });
                            titleText = `${pctText}%`;
                            content = titleText;
                          } else if (isNumericValue && isNumericColumn) {
                            const baseDecimalsForCol =
                              columnDecimalPlaces[c] ?? 2;

                            // 🔑 Only apply the global max to columns that
                            // actually have fractional digits (> 0). Integer-only
                            // columns keep their 0-decimal formatting.
                            const decimalsForCol =
                              lockNumericDecimalsToGlobalMax &&
                                baseDecimalsForCol > 0
                                ? globalMaxDecimalPlaces
                                : baseDecimalsForCol;

                            // String version for the tooltip
                            titleText = numericValue.toLocaleString("en-US", {
                              minimumFractionDigits: decimalsForCol,
                              maximumFractionDigits: decimalsForCol
                            });

                            // Visual numeric renderer (tabular-nums, right-aligned)
                            content = (
                              <DecimalAlignedNumber
                                value={numericValue}
                                decimals={decimalsForCol}
                              />
                            );
                          } else {
                            titleText = String(rawValue ?? "");
                            content = titleText;
                          }

                          return (
                            <td
                              key={c}
                              style={{
                                borderBottom: `1px solid ${rgba(
                                  theme.border,
                                  0.55
                                )}`,
                                padding: "7px 8px",
                                whiteSpace: "nowrap",
                                fontSize: 14,
                                maxWidth: columnMaxWidths[c] ?? 240,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                textAlign: isNumericColumn ? "right" : "left",
                                ...(isNumericColumn
                                  ? {
                                    // helps digits (and decimals) line up
                                    fontVariantNumeric: "tabular-nums"
                                  }
                                  : {})
                              }}
                              title={titleText || undefined}
                            >
                              {content}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Small helper to keep the "default sort column" logic tidy.
 * - When disableInitialSort is true, we avoid auto-picking a sort column
 *   so initial render preserves row order until user clicks a header.
 * - Otherwise, we pick "Model Importance" (desc) or the first column (asc).
 */
function useEffectForColumns(
  cols: string[],
  sortCol: string,
  setSortCol: (v: string) => void,
  setSortDir: (v: SortDir) => void,
  disableInitialSort: boolean
): void {
  React.useEffect(() => {
    if (!cols.length) {
      setSortCol("");
      setSortDir("asc");
      return;
    }

    if (disableInitialSort) {
      // For previews: do not auto-assign a sort column.
      // Keep current sort if it's still valid; otherwise clear.
      if (sortCol && cols.includes(sortCol)) {
        return;
      }
      setSortCol("");
      setSortDir("asc");
      return;
    }

    // Default behavior for non-preview tables (Predictions, Feature Affects).
    const preferredCol = cols.includes("Model Importance")
      ? "Model Importance"
      : cols[0];
    if (!sortCol || !cols.includes(sortCol)) {
      setSortCol(preferredCol);
      setSortDir(preferredCol === "Model Importance" ? "desc" : "asc");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cols.join("|"), disableInitialSort]);
}
