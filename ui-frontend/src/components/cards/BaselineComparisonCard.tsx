// BaselineComparisonCard.tsx
import { fmtMetric } from "../../utils/uiResults";
import { MetricRow } from "../common/MetricRow";
import { RenderStars } from "../common/RenderStars";
import { RichTooltip } from "../common/RichTooltip";
import { rightJustifiedMetric } from "../common/rightJustifiedMetric";
import { styles } from "../../theme/styles";
import type { Theme } from "../../theme/theme";

export type BaselineComparison = {
  task: "classification" | "regression" | string;
  metric: string;
  orientation: "higher_is_better" | "lower_is_better" | string | null;

  baseline_type: "majority_class" | "mean" | "median" | string | null;
  baseline_value: number | null;
  baseline_loss: number | null;

  model_value: number | null;
  model_loss: number | null;

  absolute_improvement: number | null;
  relative_improvement_percent: number | null;
  relative_improvement_stars: number | null;

  n_samples: number | null;
  n_classes?: number | null;
};

type Props = {
  theme: Theme;
  baseline: BaselineComparison | null | undefined;
  tooltips: Record<string, string>;
  display_metric: string;
};

export function BaselineComparisonCard(props: Props): JSX.Element {
  const { theme, baseline, tooltips, display_metric } = props;

  // ----- Fallback: no baseline available -----
  if (
    !baseline ||
    baseline.baseline_value == null ||
    baseline.model_value == null ||
    baseline.relative_improvement_percent == null
  ) {
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
            Comparing The Model To A Baseline
          </div>
          <div style={{ fontSize: 14, color: theme.text2, lineHeight: 1.5 }}>
            Baseline comparison is not available for this dataset. This can happen
            if the training labels are missing, empty, or the baseline metric
            cannot be computed safely.
          </div>
        </div>
      </div>
    );
  }

  const {
    metric,
    orientation,
    baseline_type,
    baseline_value,
    model_value,
    absolute_improvement,
    relative_improvement_percent,
    relative_improvement_stars,
  } = baseline;

  const fmtMaybeMetric = (v: number | null): string => {
    if (v == null || !Number.isFinite(v)) return "—";
    return fmtMetric(v, false);
  };

  const fmtPercent = (v: number | null): string =>
    v == null || !Number.isFinite(v) ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;

  const prettyBaselineType = (() => {
    return `(${baseline_type})`
    // switch (baseline_type) {
    //   case "majority_class":
    //     return "";
    //   case "mean":
    //     return "(mean)";
    //   case "median":
    //     return "(median)";
    //   case "R2-base":
    //     return "(R2-base)";
    //   default:
    //     return "";
    // }
  })();

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
          Comparing The Model To A Baseline
          {tooltips.baselineOverview ? (
            <RichTooltip html={tooltips.baselineOverview} theme={theme} />
          ) : null}
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "195px 1fr",
            rowGap: 10,
            columnGap: 20,
            fontSize: 14,
            marginTop: 12,
          }}
        >
          <MetricRow
            theme={theme}
            label="Metric Type:"
            value={rightJustifiedMetric(theme, display_metric || "—")}
            tooltipHtml={tooltips.metric}
          />

          <MetricRow
            theme={theme}
            label={`Baseline Metric ${prettyBaselineType}:`}
            value={rightJustifiedMetric(theme, fmtMaybeMetric(baseline_value))}
            tooltipHtml={tooltips.baselineType}
          />

          <MetricRow
            theme={theme}
            label="Validation Metric:"
            value={rightJustifiedMetric(theme, fmtMaybeMetric(model_value))}
            tooltipHtml={tooltips.baselineValMetric}
          />

          <MetricRow
            theme={theme}
            label="Absolute improvement:"
            value={rightJustifiedMetric(
              theme,
              fmtMaybeMetric(absolute_improvement)
            )}
            tooltipHtml={tooltips.baselineAbsolute}
          />

          <MetricRow
            theme={theme}
            label="Improvement vs Baseline:"
            value={rightJustifiedMetric(
              theme,
              fmtPercent(relative_improvement_percent)
            )}
            tooltipHtml={tooltips.baselineRelative}
            stars={<RenderStars theme={theme} stars={relative_improvement_stars} />}
          />
        </div>
      </div>
    </div>
  );
}
