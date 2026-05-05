// components/cards/Statistics.tsx
import { fmtMetric, fmt4 } from "../../utils/uiResults";
import { MetricRow } from "../common/MetricRow";
import { RenderStars } from "../common/RenderStars";
import { rightJustifiedMetric } from "../common/rightJustifiedMetric";
import { styles } from "../../theme/styles";
import type { Theme } from "../../theme/theme";

type Props = {
  theme: Theme;
  tooltips: Record<string, string>;
  display_metric: string;
  display_task: string;
  trainMetric: unknown;
  valMetric: unknown;
  ratio: unknown;
  trainMetricStars: number | null;
  valMetricStars: number | null;
  ratioStars: number | null;
};

export function StatisticsCard(props: Props): JSX.Element {
  const {
    theme,
    tooltips,
    display_metric,
    display_task,
    trainMetric,
    valMetric,
    ratio,
    trainMetricStars,
    valMetricStars,
    ratioStars,
  } = props;

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
          Model Overview
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
            label="Model Task:"
            value={rightJustifiedMetric(theme, display_task)}
            tooltipHtml={tooltips.task}
          />

          <MetricRow
            theme={theme}
            label="Metric Type:"
            value={rightJustifiedMetric(theme, display_metric)}
            tooltipHtml={tooltips.metric}
          />

          <MetricRow
            theme={theme}
            label="Training Metric:"
            value={rightJustifiedMetric(theme, fmtMetric(trainMetric))}
            tooltipHtml={tooltips.trainMetric}
            stars={<RenderStars theme={theme} stars={trainMetricStars} />}
          />

          <MetricRow
            theme={theme}
            label="Validation Metric:"
            value={rightJustifiedMetric(theme, fmtMetric(valMetric))}
            tooltipHtml={tooltips.valMetric}
            stars={<RenderStars theme={theme} stars={valMetricStars} />}
          />

          {ratio !== null && (
            <MetricRow
              theme={theme}
              label="Robustness:"
              value={rightJustifiedMetric(theme, fmt4(ratio))}
              tooltipHtml={tooltips.ratio}
              stars={<RenderStars theme={theme} stars={ratioStars} />}
            />
          )}

        </div>
      </div>
    </div>
  );
}
