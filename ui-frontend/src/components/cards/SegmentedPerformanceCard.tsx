import React from "react";
import type { Theme } from "../../theme/theme";
import { styles } from "../../theme/styles";
import { RichTooltip } from "../common/RichTooltip";

// ---- Types mirroring the Python segmented_performance payload ----

export type SegmentedConfidenceBandSegment = {
  band_label: string;
  lower: number;
  upper: number;
  count: number;
  fraction: number;
  metric_value?: number;
};

export type SegmentedConfidenceBands = {
  available: boolean;
  segments?: SegmentedConfidenceBandSegment[];
  reason?: string;
};

export type SegmentedCategoricalSegment = {
  value: string;
  count: number;
  fraction: number;
  metric_value?: number;
  target_mean?: number | null;
};

export type SegmentedCategoricalFeature = {
  feature_name: string;
  n_segments: number;
  segments: SegmentedCategoricalSegment[];
};

export type SegmentedTargetQuantileSegment = {
  target_low: number;
  target_high: number;
  count: number;
  fraction: number;
  target_mean?: number | null;
  metric_value?: number;
};

export type SegmentedTargetQuantiles = {
  available: boolean;
  segments?: SegmentedTargetQuantileSegment[];
  reason?: string;
};

export type SegmentedPerformance = {
  available: boolean;
  task: string | null;
  metric: string;
  n_samples: number;
  by_confidence_band: SegmentedConfidenceBands;
  by_target_quantile: SegmentedTargetQuantiles;
};

type Props = {
  theme: Theme;
  summary: SegmentedPerformance | null | undefined;
  tooltips: Record<string, string>;
};

// Rank label type for both views
type PerformanceRank = "weakest" | "weaker" | "better" | "best" | null;

export function SegmentedPerformanceCard(props: Props): JSX.Element | null {
  const { theme, summary, tooltips } = props;

  // Empty / unavailable state
  if (!summary || !summary.available) {
    return (
      <div style={styles.stepSectionGap(theme)}>
        <div
          style={{
            ...styles.panel(theme),
            maxWidth: 515,
          }}
        >
          <div
            style={{
              ...styles.subtleTitle(theme),
            }}
          >
            Where The Model Works Best
          </div>
          <div
            style={{
              fontSize: 14,
              color: theme.text3,
              textAlign: "center",
            }}
          >
            Segmented performance is not available for this run.
          </div>
        </div>
      </div>
    );
  }

  const {
    task,
    metric,
    by_confidence_band,
    by_target_quantile,
  } = summary;

  const isClassification = task === "classification";
  const isRegression = task === "regression";

  const fmtPercent = (v: number, decimals: number = 1): string =>
    Number.isFinite(v) ? `${(v * 100).toFixed(decimals)}%` : "—";

  const fmtNumber = (
    v: number | null | undefined,
    digits: number = 2
  ): string => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits));

  const fmtBandRange = (low: number, high: number): string => {
    const safeHigh = Math.min(high, 1); // clamp sentinel 1.01 down to 1.00
    return `${fmtPercent(low, 0)}–${fmtPercent(safeHigh, 0)}`;
  };

  const fmtTargetRange = (low: number, high: number): string =>
    `Target Range: ${fmtNumber(low, 2)} → ${fmtNumber(high, 2)}`;

  const fmtMetricMaybe = (value: number | undefined): string =>
    value == null || !Number.isFinite(value) ? "—" : value.toFixed(4);

  const hasConfidenceBands =
    !!by_confidence_band &&
    by_confidence_band.available &&
    !!by_confidence_band.segments &&
    by_confidence_band.segments.length > 0;

  const hasTargetQuantiles =
    !!by_target_quantile &&
    by_target_quantile.available &&
    !!by_target_quantile.segments &&
    by_target_quantile.segments.length > 0;

  const confidenceSegments = by_confidence_band.segments ?? [];
  const targetQuantileSegments = by_target_quantile.segments ?? [];

  const nothingToShow = !hasConfidenceBands && !hasTargetQuantiles;

  const friendlyBandLabel = (seg: SegmentedConfidenceBandSegment): string => {
    const { upper } = seg;
    if (upper <= 0.5) return "Less Confident";
    if (upper <= 0.7) return "Somewhat Confident";
    if (upper <= 0.85) return "Confident";
    return "Very Confident";
  };

  const metricLower = (metric || "").toLowerCase();
  const errorMetrics = [
    "mae",
    "mse",
    "rmse",
    "rmsle",
    "mape",
    "log_loss",
    "cross_entropy",
  ];
  const isErrorMetric = errorMetrics.includes(metricLower);

  function getPerformanceRanksForSegments<T extends { metric_value?: number }>(
    segments: T[]
  ): PerformanceRank[] {
    const n = segments.length;
    if (n === 0) return [];

    type Scored = { idx: number; value: number };
    const scored: Scored[] = [];

    segments.forEach((seg, idx) => {
      const v = seg.metric_value;
      if (typeof v === "number" && Number.isFinite(v)) {
        scored.push({ idx, value: v });
      }
    });

    if (scored.length === 0) {
      return Array<PerformanceRank>(n).fill(null);
    }

    scored.sort((a, b) =>
      isErrorMetric ? a.value - b.value : b.value - a.value
    );

    const k = scored.length;
    const ranks: PerformanceRank[] = Array<PerformanceRank>(n).fill(null);

    scored.forEach((entry, position) => {
      let label: PerformanceRank;
      if (position === 0) {
        label = "best";
      } else if (position === k - 1) {
        label = "weakest";
      } else if (position <= (k - 1) / 2) {
        label = "better";
      } else {
        label = "weaker";
      }
      ranks[entry.idx] = label;
    });

    return ranks;
  }

  const targetQuantileRanks: PerformanceRank[] = hasTargetQuantiles
    ? getPerformanceRanksForSegments(targetQuantileSegments)
    : [];

  const confidenceRanks: PerformanceRank[] = hasConfidenceBands
    ? getPerformanceRanksForSegments(confidenceSegments)
    : [];

  if (nothingToShow) {
    return (
      <div style={styles.stepSectionGap(theme)}>
        <div
          style={{
            ...styles.panel(theme),
            maxWidth: 515,
          }}
        >
          <div
            style={{
              ...styles.subtleTitle(theme),
            }}
          >
            Where The Model Works Best
          </div>
          <div
            style={{
              fontSize: 14,
              color: theme.text3,
              textAlign: "center",
            }}
          >
            Segmented performance is not available for this run.
          </div>
        </div>
      </div>
    );
  }

  const titleTooltipHtml =
    (hasConfidenceBands &&
      isClassification &&
      tooltips.segPerfConfidenceBands) ||
    (hasTargetQuantiles &&
      isRegression &&
      tooltips.segPerfTargetQuantiles) ||
    "";

  const renderPerformancePill = (rank: PerformanceRank): JSX.Element | null => {
    if (!rank) return null;

    const label =
      rank === "weakest"
        ? "Worst"
        : rank === "weaker"
          ? "Worse"
          : rank === "better"
            ? "Better"
            : "Best";

    return (
      <div
        style={{
          fontSize: 12,
          fontWeight: 500,
          padding: "2px 6px",
          borderRadius: 999,
          border: `1px solid ${theme.border2}`,
          whiteSpace: "nowrap",
          textAlign: "center",
        }}
      >
        {label}
      </div>
    );
  };

  return (
    <div style={styles.stepSectionGap(theme)}>
      <div
        style={{
          ...styles.panel(theme),
          maxWidth: 515,
        }}
      >
        {/* Title */}
        <div
          style={{
            ...styles.subtleTitle(theme),
          }}
        >
          Where The Model Works Best
          {titleTooltipHtml ? (
            <RichTooltip html={titleTooltipHtml} theme={theme} />
          ) : null}
        </div>

        {/* Confidence bands (classification) */}
        {hasConfidenceBands && isClassification && (
          <div
            style={{
              marginBottom: hasTargetQuantiles ? 16 : 0,
              paddingBottom: hasTargetQuantiles ? 12 : 0,
              borderBottom: hasTargetQuantiles
                ? `1px solid ${theme.border2}`
                : "none",
              marginTop: 12, // small top spacing similar to grid cards
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
              }}
            >
              {confidenceSegments.map((seg, idx) => {
                const rank = confidenceRanks[idx];

                return (
                  <div
                    key={seg.band_label}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 80px 60px",
                      columnGap: 8,
                      alignItems: "center",
                      padding: "6px 8px",
                      borderRadius: 6,
                      backgroundColor: theme.surface2,
                    }}
                  >
                    {/* Column 1: label + range + rows */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        fontSize: 15,
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 400,
                          color: theme.text,
                        }}
                      >
                        {friendlyBandLabel(seg)} (
                        {fmtBandRange(seg.lower, seg.upper)})
                      </div>
                      <div style={{ color: theme.text3 }}>
                        {fmtPercent(seg.fraction)} of the rows
                      </div>
                    </div>

                    {/* Column 2: metric */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-end",
                        fontSize: 15,
                        color: theme.text2,
                        fontVariantNumeric: "tabular-nums",
                        textAlign: "right",
                      }}
                    >
                      <div style={{ fontWeight: 400 }}>Metric</div>
                      <div>{fmtMetricMaybe(seg.metric_value)}</div>
                    </div>

                    {/* Column 3: pill */}
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "flex-start",
                      }}
                    >
                      {renderPerformancePill(rank)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Target quantiles (regression) */}
        {hasTargetQuantiles && isRegression && (
          <div
            style={{
              marginTop: hasConfidenceBands ? 8 : 12,
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
              }}
            >
              {targetQuantileSegments.map((seg, idx) => {
                const rank = targetQuantileRanks[idx];

                return (
                  <div
                    key={`${seg.target_low}-${seg.target_high}-${idx}`}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 80px 60px",
                      columnGap: 8,
                      alignItems: "center",
                      padding: "6px 8px",
                      borderRadius: 6,
                      backgroundColor: theme.surface2,
                    }}
                  >
                    {/* Column 1: range + rows */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        fontSize: 15,
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 400,
                          color: theme.text,
                        }}
                      >
                        {fmtTargetRange(seg.target_low, seg.target_high)}
                      </div>
                      <div style={{ color: theme.text3 }}>
                        {fmtPercent(seg.fraction)} of the rows
                      </div>
                    </div>

                    {/* Column 2: metric */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "flex-end",
                        fontSize: 15,
                        color: theme.text2,
                        fontVariantNumeric: "tabular-nums",
                        textAlign: "right",
                      }}
                    >
                      <div style={{ fontWeight: 400 }}>Metric</div>
                      <div>{fmtMetricMaybe(seg.metric_value)}</div>
                    </div>

                    {/* Column 3: pill */}
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "flex-start",
                      }}
                    >
                      {renderPerformancePill(rank)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
