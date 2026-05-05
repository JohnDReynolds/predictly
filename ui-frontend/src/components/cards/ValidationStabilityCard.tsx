import { fmtMetric } from "../../utils/uiResults";
import { MetricRow } from "../common/MetricRow";
import { RenderStars } from "../common/RenderStars";
import { RichTooltip } from "../common/RichTooltip";
import { rightJustifiedMetric } from "../common/rightJustifiedMetric";
import { styles } from "../../theme/styles";
import type { Theme } from "../../theme/theme";

export type ValidationStability = {
  n_units: number;

  mean: number | null;
  std: number | null;

  stability_stars: number | null;
  stability_message: string;

  values: number[];
};

type Props = {
  theme: Theme;
  stability: ValidationStability;
  tooltips: Record<string, string>;
  display_metric: string;
};

export function ValidationStabilityCard(props: Props): JSX.Element {
  const { theme, stability, tooltips, display_metric } = props;

  const { n_units, mean, std, stability_stars, stability_message, values } =
    stability;

  const foldCount = n_units || (values ? values.length : 0);

  const fmtMaybeMetric = (v: number | null): string =>
    v == null ? "—" : fmtMetric(v);

  // --- Range computation ---
  let rangeMinValue: number | null = null;
  let rangeMaxValue: number | null = null;

  if (values && values.length > 0) {
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    rangeMinValue = minVal;
    rangeMaxValue = maxVal;
  }

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
          {`Validation Metric Variation Across ${foldCount} Folds`}
          {tooltips.valOverview ? (
            <RichTooltip html={tooltips.valOverview} theme={theme} />
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
            label="Metric:"
            value={rightJustifiedMetric(theme, display_metric || "—")}
            tooltipHtml={tooltips.metric}
          />

          <MetricRow
            theme={theme}
            label="Range (min → max):"
            value={rightJustifiedMetric(
              theme,
              `${fmtMaybeMetric(rangeMinValue)} → ${fmtMaybeMetric(
                rangeMaxValue
              )}`
            )}
            tooltipHtml={tooltips.valRange}
          />

          <MetricRow
            theme={theme}
            label="Mean ± Std:"
            value={rightJustifiedMetric(
              theme,
              `${fmtMaybeMetric(mean)} ± ${fmtMaybeMetric(std)}`
            )}
            tooltipHtml={tooltips.valMeanStd}
          />

          <MetricRow
            theme={theme}
            label="Variation:"
            value={
              <span
                style={{
                  color: theme.text2,
                  display: "inline-flex",
                  gap: 6,
                  alignItems: "baseline",
                }}
              >
                {stability_message ? (
                  <span
                    style={{
                      fontSize: 15,
                      color: theme.text2,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {stability_message}
                  </span>
                ) : null}
              </span>
            }
            tooltipHtml={tooltips.valVariation}
            stars={<RenderStars theme={theme} stars={stability_stars} />}
          />
        </div>
      </div>
    </div>
  );
}
