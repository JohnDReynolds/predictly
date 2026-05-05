/**
 * App.tsx — Predictly Wizard UI — Auto Dark/Light + Debug Toggle
 */

import React, { useEffect, useMemo, useRef, useState } from "react";

import { RichTooltip } from "./components/common/RichTooltip";
import { SortableTableSection, getColumns } from "./components/tables/SortableTableSection";
import { UploadStep } from "./components/steps/UploadStep";
import { ParamsStep } from "./components/steps/ParamsStep";
import { PredictStep } from "./components/steps/PredictStep";
import { TopNav } from "./components/layout/TopNav";
import { FooterNav } from "./components/layout/FooterNav";
import { StatisticsCard } from "./components/cards/StatisticsCard";
import { ValidationStabilityCard, type ValidationStability } from "./components/cards/ValidationStabilityCard";
import { BaselineComparisonCard, type BaselineComparison } from "./components/cards/BaselineComparisonCard";
// import { SensitivitySummaryCard, type SensitivitySummary } from "./components/cards/SensitivitySummaryCard";
import { SegmentedPerformanceCard, type SegmentedPerformance } from "./components/cards/SegmentedPerformanceCard";


import { Theme, DARK_THEME, LIGHT_THEME, rgba } from "./theme/theme";
import { styles } from "./theme/styles";
import { usePrefersDark } from "./theme/usePrefersDark";
import { apiUrl, fetchJsonWithTimeout } from "./api/uiClient";
import { normalizeBackendPayload, normalizeRecords, fmt4 } from "./utils/uiResults";
import type { UiMessage } from "./utils/uiResults";
import { NewToThisPanel } from "./components/common/NewToThisPanel";

/** The wizard has four steps. */
type StepKey = 1 | 2 | 3 | 4;

/** Upload kinds. */
type UploadKind = "training" | "test";

/** Preview state returned by upload endpoint. */
type PreviewState = {
  description: string;
  records: Array<Record<string, unknown>>;
};

/** Meta returned after uploading the prediction file (Step 2). */
type Step2Meta = {
  /** Map from task name -> list of valid metrics for that task. */
  validTaskMetrics: Record<string, string[]>;
  uniqueColumns: string[];
  yColumnName: string;
};

/** Step 3 parameters. */
type ParamsState = {
  /** Selected task (e.g., "binary", "Regression"). */
  task: string;
  metric: string;
  uidColumnName: string; // "" allowed
};

type PageKey = "home" | "about" | "contact";

/**
 * Phase 2 status model.
 */
type TrainStatusState = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "BUSY" | "UNKNOWN";
type TrainStatusPayload = {
  status?: string;
  state: TrainStatusState;
  updated_at_epoch?: number;
  error_type?: string;
  message?: string;
  [key: string]: unknown;
};


// ----------------------
// Small helpers
// ----------------------
function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });
}


// ----------------------
// Timeouts / size limits
// ----------------------
const MAX_UPLOAD_BYTES = 4000 * 100 * 10; // 4 MB was 2 MB
const MAX_UPLOAD_MB = Math.round((MAX_UPLOAD_BYTES / 1_000_000) * 10) / 10;

const UPLOAD_TIMEOUT_MS = 180_000;

const TRAIN_ASYNC_START_TIMEOUT_MS = 300_000; // 5 mins
const TRAIN_STATUS_TIMEOUT_MS = 45_000; // 45 seconds
const TRAIN_RESULT_TIMEOUT_MS = 90_000; // 90 seconds

const POLL_INTERVAL_MS = 6_000; // Poll predicting status every 6 seconds

// JDR's stuff
const DO_DEBUG = false;
const SPEED = 0;
const SHOW_EXTENDED_RESULTS = true;


// ----------------------
// Tooltip (HTML) — closes on outside click
// ----------------------
const VALIDATION_METRIC = `The performance of the model on unseen data.`


function tooltipSamples(theme: Theme): Record<string, string> {
  return {
    baselineAbsolute: `
      <div><b>Absolute improvement over the Baseline.</b></div>
      <div style="margin-top:10px;">
        This number indicates how much better or worse the model metric is compared to the baseline metric.
        It is expressed in the same units as the metric itself.
      </div>
      <div style="margin-top:10px;">
        For <b>ROC_AUC</b>, the random baseline is <b>0.50</b>. Values above 0.50 indicate the model ranks
        positive examples higher than negatives better than random.
      </div>
    `,

    baselineOverview: `
      <div><b>Compares your model's Validation Metric against a Baseline Metric.</b></div>
      <div style="margin-top:10px;">
        A <b>baseline</b> is a simple reference model used for comparison.
        It does not learn patterns from the features. Instead, it uses a fixed or simple reference strategy,
        and its score is computed using the <b>same metric</b> as your model so the two numbers are directly comparable.
      </div>
      <div style="margin-top:10px;">
        If your model clearly outperforms the baseline, that is strong evidence that it is learning meaningful structure rather than simply guessing.
      </div>
      <div style="margin-top:10px;">
        <b>Action:</b> If your model is not significantly better than the baseline, then your training data may be noisy, too small, or missing
        important features. Refer to the <b>Training Data Health</b> section below for ideas on how to improve your training data.
      </div>
    `,

    baselineRelative: `
      <div><b>Relative improvement over the baseline.</b></div>
      <div style="margin-top:10px;">
        This number indicates how much better or worse the model metric is compared to the baseline metric
        <b>in percentage terms</b>.
      </div>
      <div style="margin-top:10px;">
        <b>Positive</b> values mean the model performs better than the baseline, while
        <b>negative</b> values mean it performs worse than the baseline.
      </div>
      <div style="margin-top:10px;">
        For most metrics, this is computed relative to the baseline value itself.
        For <b>ROC_AUC</b>, it is computed relative to the remaining headroom above the random baseline of <b>0.50</b>.
      </div>
      <div style="margin-top:10px;">
        <b>Action:</b> If your model is not significantly better than the baseline, then your training data may be noisy, too small, or missing
        important features. Refer to the <b>Training Data Health</b> section below for ideas on how to improve your training data.
      </div>
    `,

    baselineType: `
      <div><b>Baseline Metric</b></div>

      <div style="margin-top:10px;">
        The baseline uses a simple <b>reference strategy</b>. It is evaluated using the same metric
        as your model, and the value shown is the baseline's score under that metric.
      </div>

      <div style="margin-top:10px;">
        <ul style="margin-top:6px; padding-left:18px;">
          <li style="margin-bottom:6px;">
            <b>Accuracy</b>:
            predicts the <b>majority class</b> → baseline = majority-class percent.
          </li>
          <li style="margin-bottom:6px;">
            <b>Balanced Accuracy</b>:
            for a predictor that assigns the <b>same class to every row</b> → baseline = <b>1 / #classes</b>.
          </li>
          <li style="margin-bottom:6px;">
            <b>Log Loss</b>:
            predicts the <b>empirical class distribution</b> → baseline = log loss of those class probabilities.
          </li>
          <li style="margin-bottom:6px;">
            <b>PR_AUC</b>:
            uses a <b>random ranking</b> baseline → baseline = <b>positive rate</b> (fraction of rows that are positive).
          </li>
          <li style="margin-bottom:6px;">
            <b>ROC_AUC</b>:
            uses a <b>random ranking</b> baseline → baseline = <b>0.50</b>.
          </li>
          <li style="margin-bottom:6px;">
            <b>MAE</b>:
            uses the better of the <b>mean or median</b> constant predictor, usually the median → baseline = the resulting MAE.
          </li>
          <li style="margin-bottom:6px;">
            <b>MSE / RMSE</b>:
            uses the <b>mean of y</b> as the constant prediction → baseline = the resulting MSE or RMSE.
          </li>
          <li style="margin-bottom:6px;">
            <b>RMSLE</b>:
            clips negatives to 0, averages y in <b>log1p space</b>, converts back to a constant prediction, and reports the resulting RMSLE.
          </li>
          <li>
            <b>R2</b>:
            the constant-mean predictor corresponds to <b>R2 = 0</b>.
          </li>
        </ul>
      </div>
    `,

    baselineValMetric: `
      <div><b>The Model's Validation Metric.</b></div>
      <div style="margin-top:10px;">
        ${VALIDATION_METRIC}
      </div>
    `,

    dataHealthOverview: `
      <div>
        This section gives a quick health check-up for each column in your data.
        For every column, you can see the ranges, the uniqueness, and how much data is missing.
      </div>
      <div style="margin-top:10px;">
        The <b>Status</b> and <b>Messages</b> summarize any potential issues, such as having lots of
        missing values, a column that is constant or almost constant, or a target column that looks highly imbalanced.
      </div>
      <div style="margin-top:10px;">
        <b>Action:</b> Use this to identify columns that may need cleaning, transformation, or removal.
      </div>
    `,

    metric: `
      <div><b>The metric that Predictly optimizes.</b></div>
      <div style="margin-top:8px;">
        <b>Accuracy:</b> Fraction of correct predictions.
        Range: 0.0-1.0. Higher is better.
      </div>
      <div style="margin-top:8px;">
        <b>Balanced_Accuracy:</b> Accuracy averaged equally across classes.
        Range: 0.0-1.0. Higher is better.
      </div>
      <div style="margin-top:8px;">
        <b>Log_Loss:</b> Measures how well predicted probabilities match outcomes.
        Range: 0.0+. Lower is better.
      </div>
      <div style="margin-top:8px;">
        <b>MAE:</b> Mean absolute error (same units as the target).
        0.0 is perfect. Lower is better.
      </div>
      <div style="margin-top:8px;">
        <b>MSE:</b> Mean squared error.
        0.0 is perfect. Lower is better.
      </div>
      <div style="margin-top:8px;">
        <b>PR_AUC:</b> Measures how well positives are identified while avoiding false positives (precision–recall tradeoff).
        Range: baseline ≈ positive rate to 1.0. Higher is better.
      </div>
      <div style="margin-top:8px;">
        <b>R2 (R-squared):</b> Variance explained relative to predicting the mean.
        Typically 0.0-1.0 (can be negative). Higher is better.
      </div>
      <div style="margin-top:8px;">
        <b>ROC_AUC:</b> Measures how well positives are ranked above negatives.
        Range: 0.5-1.0. Higher is better.
      </div>
      <div style="margin-top:8px;">
        <b>RMSE:</b> Root mean squared error (same units as target).
        0.0 is perfect. Lower is better.
      </div>
      <div style="margin-top:8px;">
        <b>RMSLE:</b> Root mean squared logarithmic error.
        0.0 is perfect. Lower is better.
      </div>
    `,

    predictions: `
      <div><b>The Model's Predictions.</b></div>
      <div style="margin-top:10px;">
        What you have been waiting for... the final predicted target values for your Prediction File.
      </div>
    `,

    ratio: `
      <div><b>Robustness</b></div>
      <div style="margin-top:10px;">
        Robustness measures how consistently the model performs on validation data versus training data.
        It is expressed as a scale-free score between 0 and 1.
        A higher value closer to 1.0 means that the validation performance closely matches the training performance.
      </div>
    `,

    segPerfConfidenceBands: `
      <div>
        Rows are grouped by how <b>confident</b> the model is in its prediction.
        Each group collects rows where the model assigns similar confidence scores to its prediction.
        Within each confidence group, you can see how well the model actually performs and what percentage of rows fall into that group.
      </div>
      <div style="margin-top:10px;">
        In general, predictions that the model is more confident about should perform better than those it is less confident about.
      </div>
      <div style="margin-top:10px;">
        These confidences are computed from the model's internal scores using logistic or softmax normalization.
      </div>
    `,

    segPerfTargetQuantiles: `
      <div>
        Rows are grouped into 4 <b>equal ranges</b> of the target value.
      </div>
      <div style="margin-top:10px;">
        For each group, you can see how well the model actually performs.
      </div>
      <div style="margin-top:10px;">
        Use this to check whether the metric values are worse in parts of the target range where mistakes may be more costly.
      </div>
    `,

    sensitivityOverview: `
      <div><b>This section looks at the importance of each feature and how it actually moves the predictions.</b></div>
      <div style="margin-top:10px;">
        These effects are based on the training features, and are meant to give an intuitive sense of the strength of each feature
        rather than a precise causal estimate.
      </div>
      <div style="margin-top:10px;">
        Synthetic features are prefixed with “synthetic_” .  “^2” denotes that the feature is squared, and “*” denotes that the features are multiplied.
      </div>
      <div style="margin-top:10px;">
        <b>Action:</b> If you want to improve your model, refer to the <b>Training Data Health</b> section below for ideas on how to improve
        your training data, focusing on the features with a high importance and/or influence.
      </div>
    `,

    sensitivityImportance: `
      <div><b>How much the model relies on this feature.</b></div>
      <div style="margin-top:10px;">
        Features with a high importance are used frequently and/or early in the model's logic.
        The values are normalized so that all importances add up to 100%.
        A feature with 30% importance is roughly three times as important as one with 10%.
      </div>
      <div style="margin-top:10px;">
        High importance does not prove causation, but it does mean that this feature is internally important to the model.
      </div>
    `,

    sensitivityClassificationInfluence: `
      <div><b>How much the predictions change when the feature varies.</b></div>
      <div style="margin-top:10px;">
        Prediction Influence shows how much this feature typically changes the model's prediction across
        the feature's common values, excluding rare extremes.
      </div>
      <div style="margin-top:10px;">
        The values are normalized so that all features together add up to 100%. A feature with 30% influence has
        about three times the typical impact on the model's predictions as one with 10%.
      </div>
      <div style="margin-top:10px;">
        These values are based on the training data and provide an intuitive summary of how strongly each feature
        affects the model's predictions, not a causal estimate.
      </div>
    `,

    sensitivityRegressionInfluence: `
      <div><b>How much the predictions change when the feature varies.</b></div>
      <div style="margin-top:10px;">
        Prediction Influence shows how much this feature typically changes the model's prediction across
        the feature's common values, excluding rare extremes.
      </div>
      <div style="margin-top:10px;">
        Positive values mean higher feature values tend to increase the prediction, while negative values mean they tend to decrease it.
      </div>
      <div style="margin-top:10px;">
        These values are based on the training data and provide an intuitive summary of how strongly each feature
        affects the model's predictions, not a causal estimate.
      </div>
    `,

    step1Button: `
      <div><b>Upload a CSV file containing your training data.</b></div>
      <div style="margin-top:10px;">
        Each <b>row</b> should be a sample (e.g. a house or a customer), and each <b>column</b> should be a feature.
        One special column is the <b>target</b>, which is the outcome that Predictly will predict.
      </div>
    `,

    step2Button: `
      <div><b>Upload a CSV file containing your prediction data.</b></div>
      <div style="margin-top:10px;">
        This is the file for which Predictly will generate target predictions.  It is sometimes called the "test" file.
      </div>
      <div style="margin-top:10px;">
        Each <b>row</b> should be a sample (e.g. a house or a customer), and each <b>column</b> should be a feature.
        All feature columns that are in the Training File must be included.  Do <b>not</b> include the target column.
      </div>
    `,

    step4Button: `
      <div><b>Generate predicted target values for the Prediction File.</b></div>
      <div style="margin-top:10px;">
        Predictly will study the training features to learn how they can be used to predict the target value.
        It will then use what it has learned to predict the target value for each row in the Prediction File.
        This may take several minutes.
      </div>
    `,

    task: `
      <div><b>The type of model, typically referred to as the <b>task</b>.</b></div>
      <div style="margin-top:10px;">
        A <b>Classification</b> task will predict a category (e.g. Yes/No, A/B/C).
      </div>
      <div style="margin-top:10px;">
        A <b>Regression</b> task will predict a number (e.g. price, revenue).
      </div>
    `,

    trainMetric: `
      <div><b>The metric value for the training data.</b></div>
      <div style="margin-top:10px;">
        The performance of the model on the data that it was trained on.
        This is often better than the Validation Metric because the model has already seen this data.
      </div>
    `,

    uid: `
      <div><b>The column that uniquely identifies each row.</b></div>
      <div style="margin-top:10px;">
        This is an optional column like a date, an ID or a unique reference number.
        It makes it easier to match predictions back to your original data.
        For dates, it can be used to apply seasonality patterns across weeks, months, quarters and years.
      </div>
      <div style="margin-top:10px;">
        <b>Tip:</b> Put the Unique ID in the first column of your file.
      </div>
    `,

    valOverview: `
      <div><b>The variation of the Validation Metric across the different folds.</b></div>
      <div style="margin-top:10px;">
        Predictly evaluates models using <b>out-of-fold (OOF) validation</b>.
        This means that your data is split into multiple chunks or <b>folds</b>.
        Each fold will have its own metric value.
        The reported Validation Metric for the model is the mean of these multiple fold metric values.
      </div>
      <div style="margin-top:10px;">
        A lower variation of the multiple folds' metrics should give you more confidence in the model.
      </div>
    `,

    valMeanStd: `
      <div><b>Mean and Standard Deviation of Validation Metric</b></div>
      <div style="margin-top:10px;">
        The <b>mean</b> is the average Validation Metric across all folds and is indicative of the typical metric value.
      </div>
      <div style="margin-top:10px;">
        The <b>standard deviation</b> is representative of how much the Validation Metric varies from fold to fold.
        A smaller standard deviation means more stable, consistent metric values across folds.
      </div>
    `,

    valMetric: `
      <div><b>The metric value for unseen held-out data.</b></div>
      <div style="margin-top:10px;">
        The performance of the model on unseen held-out data that was not used for training.
        This is often worse than the Training Metric because the model has never seen this data.
        This is usually the more realistic metric to pay attention to.
      </div>
    `,

    valRange: `
      <div><b>Range of Validation Metrics</b></div>
      <div style="margin-top:10px;">
        Shows the smallest and largest Validation Metric across all folds (min → max).
        A smaller range means the folds are more consistent with each other.
        A larger range means some folds perform better or worse than others.
      </div>
    `,

    valVariation: `
      <div><b>The Coefficient of Variation for the Validation Metric</b></div>
      <div style="margin-top:10px;">
        Defined as <b>(standard deviation ÷ mean)</b> of the Validation Metric across folds.
        It measures relative variability rather than absolute variability.
      </div>
      <div style="margin-top:10px;">
        Closer to 0.0% indicates more stable and consistent performance across folds.
      </div>
    `,

    yColumnName: `
      <div><b>The target column that will be predicted.</b></div>
      <div style="margin-top:10px;">
        This is the column that you want Predictly to predict.
      </div>
    `,
  };
}

function renderMessageLine(msg: UiMessage | null): JSX.Element | null {
  if (!msg) return null;
  return (
    <div
      style={{
        color: msg.color,
        fontSize: 15,
        marginTop: 10,
        whiteSpace: "pre-line"
      }}
    >
      {msg.text}
    </div>
  );
}

const EXAMPLE_TRAINING_PREVIEW_ROWS: Array<Record<string, unknown>> = [
  {
    id: "C-001",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1600,
    quality: "low",
    price: 335000
  },
  {
    id: "C-002",
    bedrooms: 4,
    bathrooms: 3,
    sqft: 2100,
    quality: "high",
    price: 485000
  },
  {
    id: "C-003",
    bedrooms: 2,
    bathrooms: 1,
    sqft: 900,
    quality: "medium",
    price: 275000
  },
  {
    id: "C-004",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1300,
    quality: "high",
    price: 350000
  },
  {
    id: "C-005",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1400,
    quality: "low",
    price: 310000
  }
];

const EXAMPLE_PREDICTION_PREVIEW_ROWS: Array<Record<string, unknown>> = [
  {
    id: "C-101",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1500,
    quality: "high",
  },
  {
    id: "C-102",
    bedrooms: 4,
    bathrooms: 3,
    sqft: 2200,
    quality: "low",
  },
  {
    id: "C-103",
    bedrooms: 2,
    bathrooms: 1,
    sqft: 950,
    quality: "medium",
  },
  {
    id: "C-104",
    bedrooms: 3,
    bathrooms: 2,
    sqft: 1350,
    quality: "low",
  },
  {
    id: "C-105",
    bedrooms: 5,
    bathrooms: 3,
    sqft: 2250,
    quality: "medium",
  }
];

function renderPreviewWithUploading(
  preview: PreviewState | null,
  theme: Theme,
  kind: "training" | "test",
  isUploading: boolean
): JSX.Element | null {
  // While the file is uploading, hide the table completely
  if (isUploading) {
    return null;
  }

  // If there is no preview yet, show example data instead of nothing.
  const isExample = !preview;
  const records = isExample
    ? kind === "training"
      ? EXAMPLE_TRAINING_PREVIEW_ROWS
      : EXAMPLE_PREDICTION_PREVIEW_ROWS
    : preview?.records ?? [];

  if (!records.length) {
    return (
      <div
        style={{
          ...styles.stepSectionGap(theme),
          fontStyle: "italic",
          color: theme.text3,
          fontSize: 14
        }}
      >
        No preview rows were returned.
      </div>
    );
  }

  const rows = records.slice(0, 50);
  const title = isExample
    ? kind === "training"
      ? "" // "Sample Training Data"
      : "" // "Sample Prediction Data"
    : preview?.description || `Preview of first ${rows.length} rows`;

  return (
    <div style={{ ...styles.stepSectionGap(theme), minWidth: 0 }}>
      <SortableTableSection
        title={title}
        records={rows}
        emptyText="No preview rows were returned."
        theme={theme}
        showCopyButton={false}
        isExample={isExample}
        disableInitialSort={true}
      />
    </div>
  );
}


// ----------------------
// Main App
// ----------------------

export default function App(): JSX.Element {
  const prefersDark = usePrefersDark();
  const [themeOverride, setThemeOverride] = useState<"auto" | "dark" | "light">("auto");

  const isDark = themeOverride === "auto" ? prefersDark : themeOverride === "dark";
  const theme = isDark ? DARK_THEME : LIGHT_THEME;
  const tooltips = useMemo(() => tooltipSamples(theme), [theme]);

  const [activePage, setActivePage] = useState<PageKey>("home");
  const [activeStep, setActiveStep] = useState<StepKey>(() => getStepFromLocation());

  // Used to suppress history pushes when the step change came from a popstate event
  const isHandlingPopStateRef = useRef(false);

  const [userId, setUserId] = useState<string>("");
  const userIdRef = useRef<string>("");

  const [startupError, setStartupError] = useState<string>("");

  const [trainingUploaded, setTrainingUploaded] = useState<boolean>(false);
  const [testUploaded, setTestUploaded] = useState<boolean>(false);

  const [trainingMsg, setTrainingMsg] = useState<UiMessage | null>(null);
  const [testMsg, setTestMsg] = useState<UiMessage | null>(null);
  const [paramsMsg, setParamsMsg] = useState<UiMessage | null>(null);
  const [trainMsg, setTrainMsg] = useState<UiMessage | null>(null);

  const [trainingPreview, setTrainingPreview] = useState<PreviewState | null>(null);
  const [testPreview, setTestPreview] = useState<PreviewState | null>(null);

  const [step2Meta, setStep2Meta] = useState<Step2Meta | null>(null);
  const [params, setParams] = useState<ParamsState | null>(null);
  const [taskMetricHistory, setTaskMetricHistory] = useState<Record<string, string>>({});

  const [trainResult, setTrainResult] = useState<Record<string, unknown> | null>(null);
  const [confirmPredictAgain, setConfirmPredictAgain] = useState<boolean>(false);

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isUploadingTraining, setIsUploadingTraining] = useState<boolean>(false);
  const [isUploadingTest, setIsUploadingTest] = useState<boolean>(false);

  const [lastBackendResponse, setLastBackendResponse] = useState<unknown | null>(null);
  const [debugOpen, setDebugOpen] = useState<boolean>(false);

  const trainingInputRef = useRef<HTMLInputElement | null>(null);
  const predictionInputRef = useRef<HTMLInputElement | null>(null);

  const [trainStatus, setTrainStatus] = useState<TrainStatusPayload | null>(null);
  const [trainStatusMsg, setTrainStatusMsg] = useState<UiMessage | null>(null);

  const runTokenRef = useRef<number | null>(null);
  const pollCancelRef = useRef<boolean>(false);

  const ready = !startupError && Boolean(userId);
  const anyBusy = isTraining || isUploadingTraining || isUploadingTest;

  type UploadBindings = {
    msg: React.Dispatch<React.SetStateAction<UiMessage | null>>;
    uploaded: React.Dispatch<React.SetStateAction<boolean>>;
    preview: React.Dispatch<React.SetStateAction<PreviewState | null>>;
    inputRef: React.RefObject<HTMLInputElement>;
  };

  const uploadUi: Record<UploadKind, UploadBindings> = {
    training: {
      msg: setTrainingMsg,
      uploaded: setTrainingUploaded,
      preview: setTrainingPreview,
      inputRef: trainingInputRef
    },
    test: {
      msg: setTestMsg,
      uploaded: setTestUploaded,
      preview: setTestPreview,
      inputRef: predictionInputRef
    }
  };

  function formatFileTooLargeMessage(): UiMessage {
    return { color: theme.danger, text: `file_too_large - File cannot exceed ${MAX_UPLOAD_MB} MB.` };
  }

  function getStepFromLocation(): StepKey {
    if (typeof window === "undefined") {
      return 1;
    }

    try {
      const url = new URL(window.location.href);
      const stepParam = url.searchParams.get("step");
      const stepNum = stepParam ? parseInt(stepParam, 10) : 1;

      const clamped = Math.min(4, Math.max(1, stepNum));
      return clamped as StepKey;
    } catch {
      return 1;
    }
  }

  function initUserId(): void {
    setStartupError("");
    setUserId("");
    userIdRef.current = "";

    const cryptoObj = globalThis.crypto;
    const uid =
      cryptoObj && typeof cryptoObj.randomUUID === "function"
        ? cryptoObj.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    setUserId(uid);
    userIdRef.current = uid;
  }

  function resetWizard(): void {
    pollCancelRef.current = true;
    runTokenRef.current = null;

    setActiveStep(1);

    setTrainingUploaded(false);
    setTestUploaded(false);

    setTrainingMsg(null);
    setTestMsg(null);
    setParamsMsg(null);
    setTrainMsg(null);

    setTrainingPreview(null);
    setTestPreview(null);

    setStep2Meta(null);
    setParams(null);

    setTrainResult(null);
    setIsTraining(false);
    setConfirmPredictAgain(false);

    setIsUploadingTraining(false);
    setIsUploadingTest(false);

    setLastBackendResponse(null);
    setDebugOpen(false);

    setTrainStatus(null);
    setTrainStatusMsg(null);

    setTaskMetricHistory({});

    if (trainingInputRef.current) trainingInputRef.current.value = "";
    if (predictionInputRef.current) predictionInputRef.current.value = "";

    pollCancelRef.current = false;

    initUserId();
  }

  type WipeFrom = "training" | "test" | "params";

  function wipeDownstream(from: WipeFrom): void {
    pollCancelRef.current = true;
    runTokenRef.current = null;
    setIsTraining(false);
    setConfirmPredictAgain(false);

    if (from === "training") {
      setTestUploaded(false);
      setTestPreview(null);
      setTestMsg(null);
    }

    if (from === "training" || from === "test") {
      setStep2Meta(null);
      setParams(null);
      setParamsMsg(null);

      setTrainMsg(null);
      setTrainResult(null);

      setTrainStatus(null);
      setTrainStatusMsg(null);

      setTaskMetricHistory({});

      pollCancelRef.current = false;
      return;
    }

    // from === "params"
    setParamsMsg(null);
    setTrainMsg(null);
    setTrainResult(null);

    setTrainStatus(null);
    setTrainStatusMsg(null);

    pollCancelRef.current = false;
  }

  useEffect(() => {
    resetWizard();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep activeStep in sync when the user uses the browser Back/Forward buttons.
  useEffect(() => {
    const handlePopState = () => {
      // Mark that the next step change came from history, so we don't push another entry.
      isHandlingPopStateRef.current = true;

      const stepFromUrl = getStepFromLocation();

      // Ensure we stay on the wizard "home" page when navigating via browser history.
      setActivePage("home");
      setActiveStep(stepFromUrl);
    };

    if (typeof window !== "undefined") {
      window.addEventListener("popstate", handlePopState);
    }

    return () => {
      if (typeof window !== "undefined") {
        window.removeEventListener("popstate", handlePopState);
      }
    };
  }, []);

  // Whenever activeStep changes from inside the app, push it into the browser history
  // so the Back button will go to the previous step instead of leaving the app.
  useEffect(() => {
    if (typeof window === "undefined") return;

    // If this change came from a popstate event, we already *are* on the correct
    // history entry, so don't push a new one.
    if (isHandlingPopStateRef.current) {
      isHandlingPopStateRef.current = false;
      return;
    }

    const url = new URL(window.location.href);
    url.searchParams.set("step", String(activeStep));

    window.history.pushState({ step: activeStep }, "", url.toString());
  }, [activeStep]);

  function precheckFileSizeOrShowError(kind: UploadKind, file: File): boolean {
    const ui = uploadUi[kind];
    if (!file) return false;
    if (typeof file.size !== "number") return true;
    if (file.size <= MAX_UPLOAD_BYTES) return true;

    ui.msg(formatFileTooLargeMessage());
    return false;
  }

  async function uploadDataset(kind: UploadKind, file: File): Promise<void> {
    const ui = uploadUi[kind];

    if (!ready) {
      ui.msg({ color: theme.danger, text: "not_ready - Cannot upload: app is not ready." });
      return;
    }

    if (!precheckFileSizeOrShowError(kind, file)) return;

    if (isUploadingTraining || isUploadingTest) {
      ui.msg({ color: theme.danger, text: "ui_busy - Upload already in progress. Please wait." });
      return;
    }

    if (kind === "training") wipeDownstream("training");
    if (kind === "test") wipeDownstream("test");

    ui.uploaded(false);
    ui.preview(null);
    ui.msg({
      color: theme.text2,
      text: `Uploading ${kind === "training" ? "Training" : "Prediction"} File...`
    });

    if (kind === "training") setIsUploadingTraining(true);
    else setIsUploadingTest(true);

    const uid = userIdRef.current;

    const form = new FormData();
    form.append("user_id", uid);
    form.append("dataset_kind", kind);
    form.append("file", file);

    try {
      const resp = await fetchJsonWithTimeout(
        apiUrl("/ui/upload"),
        { method: "POST", body: form },
        UPLOAD_TIMEOUT_MS,
        theme
      );

      setLastBackendResponse(resp.payload);

      const norm = normalizeBackendPayload(resp.payload, theme);
      const resultObj = norm.resultObj;

      // If we truly have no result object at all, treat as a hard error with no preview.
      if (!resultObj) {
        ui.uploaded(false);
        ui.preview(null);
        ui.msg(norm.msg ?? { color: theme.danger, text: "upload_error - Upload failed." });
        return;
      }

      // If status is "error", we may STILL want to show preview data if it exists,
      // but we do NOT mark the upload as successful. This keeps "Next" disabled.
      if (norm.status === "error") {
        const records = normalizeRecords(resultObj.data);
        const desc =
          typeof resultObj.data_description === "string" ? resultObj.data_description : "";

        if (records.length || desc) {
          ui.preview({ description: desc, records });
        } else {
          ui.preview(null);
        }

        ui.uploaded(false);
        ui.msg(norm.msg ?? { color: theme.danger, text: "upload_error - Upload failed." });
        return;
      }

      // Success path (unchanged behavior, except for minor refactoring around resultObj reuse).
      ui.uploaded(true);
      ui.msg(norm.msg ?? null);

      const records = normalizeRecords(resultObj.data);
      const desc =
        typeof resultObj.data_description === "string" ? resultObj.data_description : "";
      ui.preview({ description: desc, records });

      if (kind === "test") {
        const validTaskMetricsRaw = resultObj.valid_task_metrics;
        const uniqueColsRaw = resultObj.unique_columns;
        const yColRaw = resultObj.y_column_name;
        const taskRaw = resultObj.task;
        const metricRaw = resultObj.metric;
        const uidRaw = resultObj.uid_column_name;

        // Primary: new valid_task_metrics shape
        const validTaskMetrics: Record<string, string[]> = {};

        if (validTaskMetricsRaw && typeof validTaskMetricsRaw === "object") {
          for (const [taskKey, metrics] of Object.entries(
            validTaskMetricsRaw as Record<string, unknown>
          )) {
            if (Array.isArray(metrics)) {
              const filtered = metrics.filter((m) => typeof m === "string") as string[];
              if (filtered.length > 0) {
                validTaskMetrics[taskKey] = filtered;
              }
            }
          }
        }

        const uniqueColumns =
          Array.isArray(uniqueColsRaw) &&
            uniqueColsRaw.every((x: unknown) => typeof x === "string")
            ? (uniqueColsRaw as string[])
            : [];

        const yColumnName = typeof yColRaw === "string" ? yColRaw : "";

        const allTaskKeys = Object.keys(validTaskMetrics);

        const defaultTaskFromJson = typeof taskRaw === "string" ? taskRaw : "";
        let initialTask = defaultTaskFromJson;
        if (!allTaskKeys.includes(initialTask)) {
          initialTask = allTaskKeys[0] ?? "";
        }

        const metricsForInitialTask = initialTask
          ? validTaskMetrics[initialTask] ?? []
          : [];

        let initialMetric = typeof metricRaw === "string" ? metricRaw : "";
        if (!metricsForInitialTask.includes(initialMetric)) {
          initialMetric = metricsForInitialTask[0] ?? "";
        }

        const defaultUid = typeof uidRaw === "string" ? uidRaw : "";

        if (!allTaskKeys.length || !yColumnName) {
          setParamsMsg({
            color: theme.danger,
            text: "schema_error - Missing metadata from prediction upload."
          });
          setStep2Meta(null);
          setParams(null);
          setTaskMetricHistory({});
        } else {
          setStep2Meta({
            validTaskMetrics,
            uniqueColumns,
            yColumnName
          });

          setParams({
            task: initialTask,
            metric: initialMetric,
            uidColumnName: defaultUid
          });

          setParamsMsg(null);

          setTaskMetricHistory(
            initialTask && initialMetric ? { [initialTask]: initialMetric } : {}
          );
        }
      }
    } finally {
      if (ui.inputRef.current) ui.inputRef.current.value = "";
      if (kind === "training") setIsUploadingTraining(false);
      else setIsUploadingTest(false);
    }
  }


  async function fetchTrainStatus(userIdForCall: string): Promise<TrainStatusPayload | null> {
    const resp = await fetchJsonWithTimeout(
      apiUrl(`/ui/train_status/${encodeURIComponent(userIdForCall)}`),
      { method: "GET" },
      TRAIN_STATUS_TIMEOUT_MS,
      theme
    );
    setLastBackendResponse(resp.payload);

    const norm = normalizeBackendPayload(resp.payload, theme);
    if (!norm.resultObj) return null;

    // Normalize state
    const stateRaw = norm.resultObj.state;
    const state: TrainStatusState =
      typeof stateRaw === "string" ? (stateRaw as TrainStatusState) : "UNKNOWN";

    // Normalize updated_at_epoch (treat 0 / non-number as "no timestamp")
    const updated = norm.resultObj.updated_at_epoch;
    const updated_at_epoch =
      typeof updated === "number" && updated > 0 ? updated : undefined;

    const msg =
      typeof norm.resultObj.message === "string" ? norm.resultObj.message : "";
    const errType =
      typeof norm.resultObj.error_type === "string" ? norm.resultObj.error_type : "";

    const statusStr =
      typeof norm.resultObj.status === "string" ? norm.resultObj.status : "";

    return {
      status: statusStr,
      state,
      updated_at_epoch,
      message: msg,
      error_type: errType
    };
  }

  async function fetchTrainResult(userIdForCall: string): Promise<Record<string, unknown> | null> {
    const resp = await fetchJsonWithTimeout(
      apiUrl(`/ui/train_result/${encodeURIComponent(userIdForCall)}`),
      { method: "GET" },
      TRAIN_RESULT_TIMEOUT_MS,
      theme
    );
    setLastBackendResponse(resp.payload);

    const norm = normalizeBackendPayload(resp.payload, theme);
    if (!norm.resultObj) return null;

    const status = typeof norm.resultObj.status === "string" ? norm.resultObj.status.toLowerCase() : "";
    if (status === "error") return norm.resultObj;

    return norm.resultObj;
  }

  async function runTrainAsync(): Promise<void> {
    if (!ready) {
      setTrainMsg({ color: theme.danger, text: "not_ready - Cannot train: app is not ready." });
      return;
    }
    if (!trainingUploaded || !testUploaded || !step2Meta || !params) {
      setTrainMsg({ color: theme.danger, text: "missing_step - Please complete steps 1–3 first." });
      return;
    }
    if (isTraining) return;

    pollCancelRef.current = false;

    setIsTraining(true);
    setTrainResult(null);

    setTrainStatus(null);
    setTrainStatusMsg(null);

    setTrainMsg({ color: theme.text2, text: "🚀 Starting async predicting... (will poll status)" });

    const uid = userIdRef.current;

    const FINALIZE_WINDOW_MS = 30_000;
    const FINALIZE_RETRY_MS = 1_000;

    try {
      const startResp = await fetchJsonWithTimeout(
        apiUrl("/ui/train_async"),
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            user_id: uid,
            task: params.task,
            metric: params.metric,
            uid_column_name: params.uidColumnName || "",
            speed: SPEED
          })
        },
        TRAIN_ASYNC_START_TIMEOUT_MS,
        theme
      );

      setLastBackendResponse(startResp.payload);

      const startNorm = normalizeBackendPayload(startResp.payload, theme);
      if (!startNorm.resultObj || startNorm.status === "error") {
        setTrainMsg(
          startNorm.msg ?? { color: theme.danger, text: "train_start_failed - Could not start training." }
        );
        return;
      }

      const startStateRaw = startNorm.resultObj.state;
      const startState = typeof startStateRaw === "string" ? (startStateRaw as TrainStatusState) : "UNKNOWN";

      const startEpochRaw = startNorm.resultObj.updated_at_epoch;
      const startEpoch = typeof startEpochRaw === "number" ? startEpochRaw : undefined;

      const startMessage = typeof startNorm.resultObj.message === "string" ? startNorm.resultObj.message : "";

      setTrainStatus({
        status: "ok",
        state: startState,
        updated_at_epoch: startEpoch,
        message: startMessage
      });

      setTrainMsg({
        color: theme.text2,
        text:
          startState === "RUNNING"
            ? "⏳ Predicting is RUNNING... polling status."
            : startState === "QUEUED"
              ? "⏳ Predicting is QUEUED... polling status."
              : `⏳ Predicting state=${startState}... polling status.`
      });

      while (!pollCancelRef.current) {
        const s = await fetchTrainStatus(uid);

        if (!s) {
          setTrainStatusMsg({ color: theme.danger, text: "status_error - Could not read predicting status." });
          await sleepMs(POLL_INTERVAL_MS);
          continue;
        }

        setTrainStatus(s);

        if (s.state === "QUEUED") {
          setTrainMsg({ color: theme.text2, text: "⏳ Predicting is QUEUED... polling status." });
          await sleepMs(POLL_INTERVAL_MS);
          continue;
        }

        if (s.state === "RUNNING") {
          setTrainMsg({ color: theme.text2, text: "⏳ Predicting is RUNNING... polling status." });
          await sleepMs(POLL_INTERVAL_MS);
          continue;
        }

        if (s.state === "FAILED") {
          const m = s.message ? `FAILED - ${s.message}` : "FAILED - Predicting failed.";
          setTrainMsg({ color: theme.danger, text: m });
          return;
        }

        if (s.state === "SUCCEEDED") {
          setTrainMsg({ color: theme.text2, text: "⏳ Finalizing... fetching results..." });

          setTrainStatus((prev) => {
            if (!prev) return prev;
            return { ...prev, state: "RUNNING" };
          });

          const deadline = Date.now() + FINALIZE_WINDOW_MS;

          while (!pollCancelRef.current && Date.now() < deadline) {
            const r = await fetchTrainResult(uid);
            if (r) {
              const rStatus = typeof r.status === "string" ? r.status.toLowerCase() : "";

              if (rStatus !== "error") {
                setTrainMsg(null);
                setTrainResult(r);

                setTrainStatus((prev) => {
                  if (!prev) return prev;
                  return { ...prev, state: "SUCCEEDED" };
                });

                return;
              }

              const et = typeof r.error_type === "string" ? r.error_type : "error";
              const mm = typeof r.message === "string" ? r.message : "Result fetch failed.";

              if (et === "not_ready") {
                await sleepMs(FINALIZE_RETRY_MS);
                continue;
              }

              // Do not show error_type to user directly
              // setTrainMsg({ color: theme.danger, text: `${et} - ${mm}` });
              setTrainMsg({ color: theme.danger, text: `${mm}` });
              setTrainResult(null);
              return;
            }

            await sleepMs(FINALIZE_RETRY_MS);
          }

          if (!pollCancelRef.current) {
            setTrainMsg({
              color: theme.danger,
              text: `finalize_timeout - Predicting finished but results were not available after ${Math.round(
                FINALIZE_WINDOW_MS / 1000
              )}s. Please reload the page or start a new run.`
            });
          }
          return;
        }

        setTrainMsg({ color: theme.text2, text: `⏳ Predicting state=${s.state}... polling status.` });
        await sleepMs(POLL_INTERVAL_MS);
      }
    } finally {
      setIsTraining(false);
      pollCancelRef.current = false;
    }
  }

  // ----------------------
  // Width logic (STEP-scoped)
  // ----------------------

  const hasTrainingPreviewTable = (trainingPreview?.records?.length ?? 0) > 0;
  const hasTestPreviewTable = (testPreview?.records?.length ?? 0) > 0;

  const hasAnyResultTable = normalizeRecords(trainResult?.y_predictions).length > 0;
  // const hasAnyResultTable =
  //   normalizeRecords(trainResult?.y_predictions).length > 0 ||
  //   (SHOW_EXTENDED_RESULTS && normalizeRecords(trainResult?.x_train_health).length > 0) ||
  //   (SHOW_EXTENDED_RESULTS && normalizeRecords(trainResult?.x_test_health).length > 0) ||
  //   (SHOW_EXTENDED_RESULTS && normalizeRecords(trainResult?.feature_importances).length > 0);

  const hasTableForActiveStep =
    activeStep === 1 ? hasTrainingPreviewTable : activeStep === 2 ? hasTestPreviewTable : activeStep === 4 ? hasAnyResultTable : false;

  const frameMaxWidth = hasTableForActiveStep ? 1100 : 560;

  function renderOutput(): JSX.Element | null {
    if (!trainResult) return null;

    const status = typeof trainResult.status === "string" ? trainResult.status : "";
    if (status.toLowerCase() === "error") return null;

    const display_metric = typeof trainResult.display_metric === "string" ? trainResult.display_metric : "";
    const display_task = typeof trainResult.display_task === "string" ? trainResult.display_task : "";
    const trainMetric = trainResult.train_metric;
    const valMetric = trainResult.validation_metric;
    const ratio = trainResult.validation_train_ratio;
    const validationStability = trainResult.validation_stability as ValidationStability;
    const baselineComparison = trainResult.baseline_comparison as BaselineComparison | null;
    //const sensitivitySummary = (trainResult as any).sensitivity_summary as SensitivitySummary | null;
    const segmentedPerformance = (trainResult as any).segmented_performance as SegmentedPerformance | null;

    // New star fields (floats)
    const trainMetricStars =
      typeof trainResult.train_metric_stars === "number"
        ? trainResult.train_metric_stars
        : null;
    const valMetricStars =
      typeof trainResult.validation_metric_stars === "number"
        ? trainResult.validation_metric_stars
        : null;
    const ratioStars =
      typeof trainResult.validation_train_ratio_stars === "number"
        ? trainResult.validation_train_ratio_stars
        : null;

    const fiRecords = normalizeRecords(trainResult.feature_importances);
    const ypRecords = normalizeRecords(trainResult.y_predictions);
    const xTrainHealthRecords = normalizeRecords(trainResult.x_train_health);
    const xTestHealthRecords = normalizeRecords(trainResult.x_test_health);

    return (
      <div>
        {/* Statistics (Overview) */}
        <div style={styles.stepSectionGap(theme)}>
          <StatisticsCard
            theme={theme}
            tooltips={tooltips}
            display_metric={display_metric}
            display_task={display_task}
            trainMetric={trainMetric}
            valMetric={valMetric}
            ratio={ratio}
            trainMetricStars={trainMetricStars}
            valMetricStars={valMetricStars}
            ratioStars={ratioStars}
          />
        </div>

        {/* Baseline Comparison */}
        {SHOW_EXTENDED_RESULTS ? (
          <div style={styles.stepSectionGap(theme)}>
            <BaselineComparisonCard
              theme={theme}
              baseline={baselineComparison}
              tooltips={tooltips}
              display_metric={display_metric}
            />
          </div>
        ) : null}

        {/* Validation Metric Variation */}
        {SHOW_EXTENDED_RESULTS ? (
          <div style={styles.stepSectionGap(theme)}>
            <ValidationStabilityCard
              theme={theme}
              stability={validationStability}
              tooltips={tooltips}
              display_metric={display_metric}
            />
          </div>
        ) : null}

        {/* WHere The Model Works Best */}
        {SHOW_EXTENDED_RESULTS && segmentedPerformance && segmentedPerformance.available ? (
          <div style={styles.stepSectionGap(theme)}>
            <SegmentedPerformanceCard
              theme={theme}
              summary={segmentedPerformance}
              tooltips={tooltips}
            />
          </div>
        ) : null}

        {/* Classification Feature Effects */}
        {display_task === "Classification" ? (
          <div
          >
            <SortableTableSection
              title="Feature Effects"
              records={fiRecords}
              emptyText="Feature Effects could not be determined."
              theme={theme}
              percentageColumns={["Model Importance", "Prediction Influence"]}
              tooltips={{
                tableOverview: tooltips.sensitivityOverview,
                "column:Model Importance": tooltips.sensitivityImportance,
                "column:Prediction Influence": tooltips.sensitivityClassificationInfluence,
              }}
            />
          </div>
        ) : null}

        {/* Regression Feature Effects */}
        {display_task === "Regression" ? (
          <div
          >
            <SortableTableSection
              title="Feature Effects"
              records={fiRecords}
              emptyText="Feature Effects could not be determined."
              theme={theme}
              percentageColumns={["Model Importance"]}
              tooltips={{
                tableOverview: tooltips.sensitivityOverview,
                "column:Model Importance": tooltips.sensitivityImportance,
                "column:Prediction Influence": tooltips.sensitivityRegressionInfluence,
              }}
            />
          </div>
        ) : null}

        {/* Training Data Health */}
        <div style={styles.stepSectionGap(theme)}>
          <SortableTableSection
            title="Training Data Health"
            records={xTrainHealthRecords}
            emptyText="No training data health information returned."
            theme={theme}
            disableInitialSort={true}
            tooltips={{ tableOverview: tooltips.dataHealthOverview }}
            percentageColumns={[
              "Majority_Pct",
              "Minority_Pct",
              "Missing_Pct",
              "Most_Frequent_Pct",
              "Unique_Pct"
            ]}
            columnMaxWidths={{ Messages: 315, }}
          />
        </div>

        {/* Prediction Data Health */}
        <div style={styles.stepSectionGap(theme)}>
          <SortableTableSection
            title="Prediction Data Health"
            records={xTestHealthRecords}
            emptyText="No prediction data health information returned."
            theme={theme}
            disableInitialSort={true}
            tooltips={{ tableOverview: tooltips.dataHealthOverview }}
            percentageColumns={[
              "Majority_Pct",
              "Minority_Pct",
              "Missing_Pct",
              "Most_Frequent_Pct",
              "Unique_Pct"
            ]}
            columnMaxWidths={{ Messages: 315, }}
          />
        </div>


        {/* Predictions */}
        <div style={styles.stepSectionGap(theme)}>
          <SortableTableSection
            title="Predictions"
            records={ypRecords}
            emptyText="No predictions returned."
            theme={theme}
            disableInitialSort={true}
            tooltips={{ tableOverview: tooltips.predictions }}
            lockNumericDecimalsToGlobalMax={true}
          />
        </div>

      </div>
    );
  }

  function renderHome(): JSX.Element | null {
    if (startupError) {
      return <div style={{ color: theme.danger, fontSize: 15 }}>{startupError}</div>;
    }

    // ----------------------
    // About page
    // ----------------------
    if (activePage === "about") {
      return (
        <div style={{ marginTop: 20, color: theme.text2 }}>
          <h2 style={{ marginTop: 0, fontSize: 19, color: theme.text }}>About</h2>

          <div style={{ ...styles.panel(theme), marginBottom: 20 }}>
            <h3
              style={{
                marginTop: 0,
                marginBottom: 0,
                fontSize: 16,
                color: theme.text,
              }}
            >
              Predictly at a glance
            </h3>

            <ul style={{ marginTop: 8, marginBottom: 0 }}>
              <li style={{ marginBottom: 6 }}>Upload your tabular data</li>
              <li style={{ marginBottom: 6 }}>Choose what you want to predict</li>
              <li style={{ marginBottom: 0 }}>Get solid predictions and analytics in minutes</li>
            </ul>

          </div>

          <div style={{ color: theme.text2, fontSize: 15, lineHeight: 1.6 }}>

            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>What Predictly does</h3>

            <p style={{ marginBottom: 2 }}>
              Predictly implements the most common tabular prediction tasks:
            </p>
            <ul style={{ marginTop: 0 }}>
              <li><b>Regression</b> — predicting a number.</li>
              <li><b>Binary Classification</b> — predicting 2 categories like True/False.</li>
              <li><b>Multiclass Classification</b> — predicting from many categories like A/B/C.</li>
            </ul>

            <p style={{ marginBottom: 2 }}>
              Predictly works with <b>tabular data</b>, the kind found in CSV files and spreadsheets.
            </p>
            <ul style={{ marginTop: 0 }}>
              <li>Each row represents one sample (e.g. a house or a customer).</li>
              <li>Each column represents a feature.</li>
              <li>One column is the target — the value that you want Predictly to predict.</li>
            </ul>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>Who is Predictly for?</h3>

            <p style={{ marginBottom: 2 }}>
              Predictly is designed for:
            </p>
            <ul style={{ marginTop: 0 }}>
              <li>Product and operations teams who want to test out ideas.</li>
              <li>Engineers who want to prototype and get fast, solid results.</li>
              <li>Anyone who wants predictions and analytics without coding.</li>
            </ul>

            <p style={{ marginBottom: 2 }}>
              It is especially useful when:
            </p>
            <ul style={{ marginTop: 0 }}>
              <li>You want results quickly.</li>
              <li>You value consistency and reliability.</li>
              <li>You want transparency instead of black boxes.</li>
            </ul>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>What Predictly is not</h3>

            <p style={{ marginBottom: 2 }}>
              Predictly does not try to be everything.
            </p>

            <ul style={{ marginTop: 0 }}>
              <li>It is not a deep-learning research platform.</li>
              <li>It is not optimized for large datasets.</li>
              <li>It is not a replacement for custom ML engineering.</li>
            </ul>

            <p>
              Predictly focuses on the most common, practical tabular problems — and does them well.
            </p>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>Designed for simplicity</h3>

            <p style={{ marginBottom: 2 }}>
              Predictly is intentionally simple.  Rather than exposing multiple tuning knobs, it focuses on:
            </p>
            <ul style={{ marginTop: 0 }}>
              <li>Sensible defaults.</li>
              <li>Clear steps.</li>
              <li>Strong validation.</li>
            </ul>

            <p>This makes it easy to get useful results without needing to be a machine-learning expert.</p>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, marginBottom: 6, fontSize: 16, color: theme.text }}>Predictly, behind the scenes</h3>
            <ul style={{ marginTop: 0 }}>
              <li>Flags and imputes missing values.</li>
              <li>Reduces the impact of outliers.</li>
              <li>Generates polynomial features from high-impact features.</li>
              <li>Encodes and scales features automatically.</li>
              <li>Balances model complexity to avoid underfitting and overfitting.</li>
            </ul>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>Out-of-Fold validation</h3>

            <p>
              Predictly evaluates models using <b>out-of-fold (OOF) validation</b>.
            </p>

            <p>Your data is split into multiple parts. Models are trained on some parts and tested on others.</p>

            <p>
              This provides a more realistic picture of how a model will perform on unseen data, not just the data it has
              already seen.
            </p>


            <hr style={{ border: "none", borderTop: `1px solid ${theme.border}`, margin: "16px 0" }} />
            <h3 style={{ marginTop: 0, fontSize: 16, color: theme.text }}>Analytics</h3>

            <p style={{ marginBottom: 2 }}>
              After modeling and traning, Predictly shows several key outputs:
            </p>
            <ul style={{ marginTop: 0 }}>
              <li>
                <b>Training Metric</b>: How well the model fits the data it trained on.
              </li>
              <li>
                <b>Validation Metric</b>: {VALIDATION_METRIC}
              </li>
              <li>
                <b>Robustness</b>: Measures how robust the model's performance is on unseen data.
              </li>
              <li>
                <b>Comparing The Model To A Baseline</b>: How well the model performs over a naive baseline.
              </li>
              <li>
                <b>Model Variation</b>: The variation of the model across different folds.
              </li>
              <li>
                <b>Where The Model Works Best</b>: How well the model performs in different buckets/groups of data.
              </li>
              <li>
                <b>Feature Effects</b>: The importance of each feature and how it actually moves the predictions.
              </li>
              <li>
                <b>Data Health</b>: Highlights data issues that may need cleaning, transformation, or removal before creating your model.
              </li>
              <li>
                <b>Predictions</b>: The final predicted target values for your prediction file.
              </li>
            </ul>

            <p>These results are designed to be easy to scan, compare, and reason about.</p>

          </div>
        </div>
      );
    }

    // ----------------------
    // Contact page
    // ----------------------
    if (activePage === "contact") {
      return (
        <div style={{ marginTop: 20, color: theme.text2 }}>
          <h2 style={{ marginTop: 0, fontSize: 19, color: theme.text }}>Contact</h2>
          <p style={{ color: theme.text2, fontSize: 15 }}>
            E-mail questions, comments, or issues to{" "}
            <a href="mailto:predictly.cloud@gmail.com" style={{ color: theme.link }}>
              predictly.cloud@gmail.com
            </a>.
          </p>
          {/* <p>
            Please include your reference ID:&nbsp;
            <code style={{ color: theme.text }}>{userId}</code>
          </p> */}
        </div>
      );
    }

    // ----------------------
    // Home page (wizard)
    // ----------------------
    const showWizard = activePage === "home";

    if (!showWizard) {
      // Fallback, but in practice we’ve covered all pages above.
      return null;
    }

    const canBack = !anyBusy && activeStep > 1;
    const canNext =
      !anyBusy &&
      ready &&
      ((activeStep === 1 && trainingUploaded) ||
        (activeStep === 2 && testUploaded) ||
        (activeStep === 3 && testUploaded && step2Meta !== null && params !== null) ||
        activeStep === 4);

    const canTrain =
      !anyBusy && trainingUploaded && testUploaded && step2Meta !== null && params !== null;

    const canBackFromStep4 = !anyBusy;

    return (
      <>
        <div style={{ marginTop: 18 }}>
          {activeStep === 1 ? <NewToThisPanel theme={theme} /> : null}

          {activeStep === 1 ? (
            <UploadStep
              heading="1. Training File"
              buttonText="Upload Training CSV File"
              buttonTooltipHtml={tooltips.step1Button}
              disabled={!ready || anyBusy}
              inputRef={trainingInputRef}
              message={trainingMsg}
              preview={trainingPreview}
              onUpload={(file) => void uploadDataset("training", file)}
              theme={theme}
              actionButtonStyle={styles.primaryButton}
              renderMessageLine={renderMessageLine}
              renderPreview={(preview, themeArg) =>
                renderPreviewWithUploading(preview, themeArg, "training", isUploadingTraining)
              }
            />
          ) : null}

          {activeStep === 2 ? (
            <UploadStep
              heading="2. Prediction File"
              buttonText="Upload Prediction CSV File"
              buttonTooltipHtml={tooltips.step2Button}
              disabled={!ready || anyBusy || !trainingUploaded}
              inputRef={predictionInputRef}
              message={testMsg}
              preview={testPreview}
              onUpload={(file) => void uploadDataset("test", file)}
              theme={theme}
              actionButtonStyle={styles.primaryButton}
              renderMessageLine={renderMessageLine}
              renderPreview={(preview, themeArg) =>
                renderPreviewWithUploading(preview, themeArg, "test", isUploadingTest)
              }
            />
          ) : null}

          {activeStep === 3 ? (
            <ParamsStep
              theme={theme}
              tooltips={tooltips}
              anyBusy={anyBusy}
              step2Meta={step2Meta}
              params={params}
              paramsMsg={paramsMsg}
              onTaskChange={(task) => {
                if (!step2Meta) return;

                const allMetricsForTask = step2Meta.validTaskMetrics[task] ?? [];
                const rememberedMetric = taskMetricHistory[task];

                const nextMetric =
                  rememberedMetric && allMetricsForTask.includes(rememberedMetric)
                    ? rememberedMetric
                    : allMetricsForTask[0] ?? "";

                setParams((prev) =>
                  prev ? { ...prev, task, metric: nextMetric } : prev
                );

                if (nextMetric) {
                  setTaskMetricHistory((prev) => ({ ...prev, [task]: nextMetric }));
                }

                wipeDownstream("params");
              }}
              onMetricChange={(metric) => {
                const currentTask = params?.task ?? "";
                setParams((pp) => (pp ? { ...pp, metric } : pp));

                if (currentTask) {
                  setTaskMetricHistory((prev) => ({ ...prev, [currentTask]: metric }));
                }

                wipeDownstream("params");
              }}
              onUidColumnChange={(uidColumnName) => {
                setParams((pp) => (pp ? { ...pp, uidColumnName } : pp));
                wipeDownstream("params");
              }}
              renderMessageLine={renderMessageLine}
            />
          ) : null}

          {activeStep === 4 ? (
            <PredictStep
              theme={theme}
              tooltips={tooltips}
              isTraining={isTraining}
              isDebug={DO_DEBUG}
              canTrain={canTrain}
              canInlineBack={canBackFromStep4}
              showInlineBack={hasAnyResultTable}
              trainMsg={trainMsg}
              trainStatus={trainStatus}
              trainStatusMsg={trainStatusMsg}
              onPredict={() => {
                if (!canTrain) return;

                // Has there been a completed run (success OR failure)?
                const lastRunFinished =
                  trainStatus?.state === "SUCCEEDED" || trainStatus?.state === "FAILED";

                // If we already finished a run, require a second click to confirm re-running.
                if (lastRunFinished && !confirmPredictAgain) {
                  setTrainMsg({
                    color: isDark ? "#f0c674" : "#7b6000",
                    text: "This will start a new Predicting job.  Click Predict again to confirm."
                  });
                  setConfirmPredictAgain(true);
                  return;
                }

                // User confirmed (or there was no prior finished run): clear the flag and go.
                setConfirmPredictAgain(false);
                void runTrainAsync();
              }}

              onBackToParams={() => {
                if (!canBackFromStep4) return;
                setActiveStep(3);
              }}
              renderMessageLine={renderMessageLine}
              renderOutput={renderOutput}
              actionButtonStyle={styles.primaryButton}
              backButtonStyle={styles.navButton}
            />
          ) : null}

          <div style={{ marginTop: 40 }}>
            <FooterNav
              theme={theme}
              activeStep={activeStep}
              canBack={canBack}
              canNext={canNext}
              onBack={() => {
                setActiveStep((s) => (s > 1 ? ((s - 1) as StepKey) : s));
              }}
              onNext={() => {
                setActiveStep((s) => (s < 4 ? ((s + 1) as StepKey) : s));
              }}
              navPrimaryButtonStyle={styles.navButton}
            />
          </div>

          {DO_DEBUG && debugOpen ? (
            <div style={{ marginTop: 14 }}>
              <div
                style={{
                  fontWeight: 900,
                  marginBottom: 6,
                  fontSize: 14,
                  color: theme.text
                }}
              >
                Debug
              </div>
              <pre
                style={{
                  background: isDark ? theme.surface3 : "#f3f5ff",
                  color: isDark ? theme.text2 : "#0f172a",
                  padding: 12,
                  borderRadius: 12,
                  overflow: "auto",
                  maxHeight: 360,
                  fontSize: 12,
                  border: `1px solid ${theme.border}`
                }}
              >
                {JSON.stringify(lastBackendResponse ?? {}, null, 2)}
              </pre>
            </div>
          ) : null}
        </div>
      </>
    );
  }


  // ----------------------
  // Debug-only manual theme toggle control
  // ----------------------

  function renderThemeToggle(): JSX.Element {
    const base: React.CSSProperties = {
      display: "inline-flex",
      border: `1px solid ${theme.border2}`,
      borderRadius: 10,
      overflow: "hidden",
      background: theme.surface2
    };

    const btn: React.CSSProperties = {
      padding: "6px 10px",
      fontSize: 13,
      fontWeight: 750,
      border: "none",
      cursor: "pointer"
    };

    const activeStyle: React.CSSProperties = {
      background: `linear-gradient(180deg, ${theme.accent}, ${theme.accent2})`,
      color: theme.onAccent
    };

    const inactiveStyle: React.CSSProperties = {
      background: theme.surface2,
      color: theme.text2
    };

    const darkActive = themeOverride === "dark";
    const lightActive = themeOverride === "light";

    return (
      <div style={base} aria-label="Theme toggle">
        <button
          type="button"
          onClick={() => setThemeOverride("dark")}
          style={{ ...btn, ...(darkActive ? activeStyle : inactiveStyle) }}
          title="Force Dark"
        >
          Dark
        </button>
        <button
          type="button"
          onClick={() => setThemeOverride("light")}
          style={{ ...btn, ...(lightActive ? activeStyle : inactiveStyle) }}
          title="Force Light"
        >
          Light
        </button>
      </div>
    );
  }

  // ----------------------
  // Top-level render
  // ----------------------

  return (
    <div style={{ minHeight: "100vh", background: theme.bg, paddingBottom: 28 }}>
      <style>
        {`
          html, body, #root {
            height: 100%;
            margin: 0;
            background: ${theme.bg};
          }
        `}
      </style>

      <div
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: 18,
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif',
          color: theme.text,
          fontSize: 15
        }}
      >

        <div style={{ marginBottom: 32 }}>
          <TopNav
            theme={theme}
            isDark={isDark}
            activePage={activePage}
            onPageChange={setActivePage}
            navLinkStyle={styles.navLink}
          />
        </div>

        <div style={{ marginTop: 14 }}>
          <div
            style={{
              ...styles.card(theme),

              maxWidth: frameMaxWidth,
              marginLeft: "auto",
              marginRight: "auto",

              boxShadow: `0 18px 60px ${rgba("#000000", isDark ? 0.5 : 0.18)}`
            }}
          >
            {DO_DEBUG ? (
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <div style={{ fontSize: 14, color: theme.text2 }}>
                  user_id: <code style={{ fontSize: 13, color: theme.text }}>{userId || "(initializing...)"}</code>
                  {themeOverride === "auto" ? (
                    <span style={{ color: theme.text3, marginLeft: 10, fontSize: 12 }}>(auto theme)</span>
                  ) : null}
                </div>

                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  {renderThemeToggle()}

                  <button
                    type="button"
                    onClick={() => setDebugOpen((x) => !x)}
                    disabled={false}
                    style={styles.compactButton(theme, false)}
                  >
                    {debugOpen ? "Hide Debug" : "Show Debug"}
                  </button>
                </div>
              </div>
            ) : null}

            {renderHome()}
          </div>
        </div>
      </div>
    </div>
  );
}


// ----------------------
// baselineOverview: `
//   <div><b>Compares your model's Validation Metric against a Baseline Metric.</b></div>
//   <div style="margin-top:6px;">
//     A <b>baseline</b> is a very simple model that does something "naive but honest",
//     such as always predicting the most common class or using a single average value.
//     It gives you a floor of how well you can do without any real modeling.
//   </div>
//   <div style="margin-top:10px;">
//     For a Classification model, the baseline uses the most frequent class in the training data.
//     This most frequent class is called the <b>majority class</b>.
//   </div>
//   <div style="margin-top:10px;">
//     For a Regression model, the baseline uses a single constant value that is either the overall <b>mean</b> or <b>median</b> of the training target values.
//   </div>
//   <div style="margin-top:10px;">
//     If your model clearly outperforms the baseline, then that is a strong signal that your model is learning useful patterns rather than noise.
//   </div>
//   <div style="margin-top:10px;">
//     <b>Action:</b> If your model cannot beat the baseline, then your training data may be noisy, too small, or missing important features.
//     Refer to the <b>Training Data Health</b> section below for ideas on how to improve your training data.
//   </div>
// `,

// baselineOverview: `
//   <div><b>Compares your model's Validation Metric against a Baseline Metric.</b></div>

//   <div style="margin-top:6px;">
//     A <b>baseline</b> is a simple constant-prediction model used as a reference.
//     Its score is computed using the <b>same metric</b> as your model,
//     so the two numbers are directly comparable.
//   </div>

//   <div style="margin-top:10px;">
//     For <b>Classification</b>, the baseline always predicts the most frequent class
//     in the training data (the <b>majority class</b>), and its metric is computed
//     from those constant predictions.
//   </div>

//   <div style="margin-top:10px;">
//     For <b>Regression</b>, the baseline always predicts a single constant value:
//     <ul style="margin-top:6px; padding-left:18px;">
//       <li>
//         For <b>MSE / RMSE</b>, it predicts the <b>mean</b> of the training targets.
//         The baseline metric is then the MSE (or RMSE) of those constant predictions.
//       </li>
//       <li>
//         For <b>MAE</b>, it predicts the <b>median</b> of the training targets,
//         because the median minimizes absolute error.
//       </li>
//     </ul>
//   </div>

//   <div style="margin-top:10px;">
//     If your model clearly outperforms the baseline, that is strong evidence that
//     it is learning meaningful structure rather than simply guessing.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Action:</b> If your model cannot beat the baseline, your data may be noisy,
//     too small, or missing important features. Review the <b>Training Data Health</b>
//     section for guidance.
//   </div>
// `,

// <div style="margin-top:10px;">
//   For <b>Classification</b>, the baseline ignores features and predicts a constant value:
//   <ul style="margin-top:6px; padding-left:18px;">
//     <li>
//       For <b>Accuracy / Balanced_Accuracy</b>, it predicts the
//       <b>most frequent class</b> in the training data (called the "majority class").
//     </li>
//     <li>
//       For <b>Log_Loss</b>, it predicts constant <b>class probabilities</b>
//       equal to the observed class frequencies in the training data.
//     </li>
//   </ul>
//   The baseline metric is then computed from those constant predictions.
// </div>

// <div style="margin-top:10px;">
//   For <b>Regression</b>, the baseline predicts a single constant value:
//   <ul style="margin-top:6px; padding-left:18px;">
//     <li>
//       For <b>MSE / RMSE / R2</b>, it predicts the <b>mean</b> of the training targets.
//     </li>
//     <li>
//       For <b>MAE</b>, it predicts the <b>median</b> of the training targets,
//       because the median minimizes absolute error.
//     </li>
//     <li>
//       For <b>RMSLE</b>, it predicts the <b>mean in log space</b>
//       (equivalent to a constant multiplicative prediction).
//     </li>
//   </ul>
//   The baseline metric is then computed from those constant predictions.
// </div>

// baselineType: `
//   <div style="margin-top:6px;">
//     <b>Baseline Metric</b>
//   </div>

//   <div style="margin-top:10px;">
//     The baseline is a simple constant-prediction model used as a reference.
//     Its score is computed using the <b>same metric</b> as the trained model.
//   </div>

//   <div style="margin-top:10px;">
//     For <b>Classification</b>, the baseline always predicts the most frequent class
//     in the training data (the <b>majority class</b>), and its metric is computed
//     from those constant predictions.
//   </div>

//   <div style="margin-top:10px;">
//     For <b>Regression</b>:
//     <ul style="margin-top:6px; padding-left:18px;">
//       <li>
//         For <b>MSE / RMSE</b>, the baseline always predicts the <b>mean</b> of the training targets.
//         The baseline metric is then the MSE (or RMSE) of those constant predictions.
//         For MSE, this value equals the variance of the target.
//       </li>
//       <li>
//         For <b>MAE</b>, the baseline always predicts the <b>median</b> of the training targets,
//         because the median minimizes absolute error.
//       </li>
//       <li>
//         For <b>R2</b> (R-squared), predicting the mean corresponds to <b>R2 = 0.0</b>.
//         Positive R2 values mean the model explains more variation than this baseline,
//         while negative values mean it performs worse.
//       </li>
//     </ul>
//   </div>
// `,

// baselineType: `
//   <div style="margin-top:6px;">
//     <b>Baseline's Metric</b>
//   </div>

//   <div style="margin-top:10px;">
//     For a <b>Classification</b> model, the baseline uses the most frequent class in the training data.
//     This most frequent class is called the <b>majority class</b>.
//   </div>

//   <div style="margin-top:10px;">
//     For most <b>Regression</b> models, the baseline uses a single constant value that is either the
//     overall <b>mean</b> or <b>median</b> of the training targets.
//   </div>

//   <div style="margin-top:10px;">
//     For <b>R2</b> (R-squared), the baseline represents a model that explains none of the variation in the target
//     (R2 = 0.0).  This is equivalent to always predicting the <b>mean</b>.
//     Positive R2 values mean the model improves on this baseline, while negative values mean it performs worse.
//   </div>
// `,


// baselineType: `
//   <div style="margin-top:6px;">
//     <b>Baseline's Metric</b>
//   </div>
//   <div style="margin-top:10px;">
//     For a Classification model, the baseline uses the most frequent class in the training data.
//     This most frequent class is called the <b>majority class</b>.
//   </div>
//   <div style="margin-top:10px;">
//     For a Regression model, the baseline uses a single constant value that is either the overall mean or median of the training targets.
//   </div>
// `,

// baselineModel: `
//   <div><b>Model's Validation Metric</b></div>
// `,

// const ROBUSTNESS_LOW_EXPLANATION_JSX = (
//   <>
//     A value below 0.90 can indicate that the model is <b>underfitted</b>, which means that the model does not learn enough to really understand the data - like skimming a book.
//   </>
// );
// const ROBUSTNESS_HIGH_EXPLANATION_JSX = (
//   <>
//     A value above 1.20 can indicate that the model is <b>overfitted</b>, which means that the model learns the training data <b>too</b> well and struggles with unseen data - like memorizing the answers.
//   </>
// );


// metric: `
//   <div><b>The metric that Predictly optimizes.</b></div>

//   <div style="margin-top:6px;">
//     <b>Accuracy:</b> Fraction of correct predictions.
//     Use when classes are fairly balanced and all mistakes matter equally.
//     Range: 0.0–1.0. Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Balanced_Accuracy:</b> Average of per-class accuracies.
//     Use when classes are imbalanced and rare classes matter.
//     Range: 0.0–1.0. Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Log_Loss:</b> Measures how well predicted probabilities match the true labels.
//     Use for classification when probability quality (not just the winning class) matters.
//     Typical values: 0.0 is perfect; values are usually between ~0.1 and 1.5. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>MAE:</b> Mean absolute error (average absolute difference).
//     Use when you want a simple average error in the same units as the target.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>MSE:</b> Mean squared error (average squared difference).
//     Use when large mistakes should be penalized more heavily than small ones.
//     0.0 is perfect; units are the square of the target. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>R2 (R-squared):</b> Share of target variation explained by the model vs the mean.
//     Use as a general “goodness of fit” for regression.
//     Typical range: 0.0–1.0 (can be negative if worse than predicting the mean). Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>RMSE:</b> Root mean squared error (MAE where big errors count extra).
//     Use when large mistakes are especially costly.
//     0.0 is perfect; units match the target. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>RMSLE:</b> Root mean squared logarithmic error.
//     Use when relative (percentage-like) errors matter more than absolute differences.
//     0.0 is perfect; focuses on ratios more than raw differences. Lower is better.
//   </div>
// `,

// metric: `
//   <div><b>The metric that Predictly optimizes.</b></div>

//   <div style="margin-top:6px;">
//     <b>Accuracy:</b> Fraction of correct predictions.
//     Best for balanced classes where all mistakes matter equally.
//     Range: 0.0–1.0. Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Balanced_Accuracy:</b> Average accuracy across classes.
//     Best for imbalanced datasets where rare classes matter.
//     Range: 0.0–1.0. Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Log_Loss:</b> Measures how well predicted probabilities match true outcomes.
//     Use when probability quality matters.
//     0.0 is perfect; typically ~0.1–1.5. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>MAE:</b> Average absolute error (same units as target).
//     Simple, easy-to-interpret error.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>MSE:</b> Mean squared error.
//     Penalizes large errors more; squared units.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>R2 (R-squared):</b> Share of variance explained vs predicting the mean.
//     General regression goodness-of-fit.
//     Usually 0.0–1.0 (can be negative). Higher is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>RMSE:</b> Root mean squared error.
//     Penalizes large errors more; same units as target.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:10px;">
//     <b>RMSLE:</b> Root mean squared logarithmic error.
//     Focuses on relative (percentage-like) differences.
//     0.0 is perfect. Lower is better.
//   </div>
// `,


// baselineOverview: `
//   <div><b>Compares your model's Validation Metric against a Baseline Metric.</b></div>

//   <div style="margin-top:10px;">
//     A <b>baseline</b> is a simple constant-prediction model used as a reference.
//     Its predictions are fixed (no feature learning), and its score is computed
//     using the <b>same metric</b> as your model so the two numbers are directly comparable.
//   </div>

//   <div style="margin-top:10px;">
//     If your model clearly outperforms the baseline, that is strong evidence
//     that it is learning meaningful structure rather than simply guessing.
//   </div>

//   <div style="margin-top:10px;">
//     <b>Action:</b> If your model is not significantly better than the baseline, then your training data may be noisy, too small, or missing
//     important features.  Refer to the <b>Training Data Health</b> section below for ideas on how to improve your training data.
//   </div>
// `,


// baselineRelative: `
//   <div><b>Relative improvement over the baseline.</b></div>
//   <div style="margin-top:10px;">
//     This number indicates how much better or worse the model metric is compared to the baseline metric
//     <b>in percentage terms</b>.
//   </div>
//   <div style="margin-top:10px;">
//     The sign reflects the metric direction:
//     positive values indicate improvement for higher-is-better metrics,
//     while negative values indicate improvement for lower-is-better metrics.
//   </div>
//   <div style="margin-top:10px;">
//     <b>Action:</b> If your model is not significantly better than the baseline, then your training data may be noisy, too small, or missing
//     important features. Refer to the <b>Training Data Health</b> section below for ideas on how to improve your training data.
//   </div>
// `,

// baselineType: `
//   <div><b>Baseline Metric</b></div>

//   <div style="margin-top:10px;">
//     The baseline uses a simple <b>constant predictor</b> as a reference. It is evaluated using the same metric
//     as your model, and the value shown is the baseline's score for that constant prediction.
//   </div>

//   <div style="margin-top:10px;">
//     <ul style="margin-top:6px; padding-left:18px;">
//       <li style="margin-bottom:6px;">
//         <b>Accuracy</b>:
//         predicts the <b>majority class</b> → baseline = majority-class percent.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>Balanced Accuracy</b>:
//         for a predictor that assigns the <b>same class to every row</b> → baseline = <b>1 / #classes</b>.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>Log Loss</b>:
//         predicts the <b>empirical class distribution</b> → baseline = log loss of those class probabilities.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>MAE</b>:
//         uses the better of the <b>mean or median</b> constant predictor, usually the median → baseline = the resulting MAE.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>MSE / RMSE</b>:
//         uses the <b>mean of y</b> as the constant prediction → baseline = the resulting MSE or RMSE.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>RMSLE</b>:
//         clips negatives to 0, averages y in <b>log1p space</b>, converts back to a constant prediction, and reports the resulting RMSLE.
//       </li>
//       <li>
//         <b>R2</b>:
//         the constant-mean predictor corresponds to <b>R2 = 0</b>.
//       </li>
//     </ul>
//   </div>
// `,

// baselineType: `
//   <div><b>Baseline Metric</b></div>

//   <div style="margin-top:10px;">
//     The baseline uses a very simple <b>reference strategy</b>. It is evaluated using the same metric
//     as your model, and the value shown is the baseline's score under that metric.
//   </div>

//   <div style="margin-top:10px;">
//     <ul style="margin-top:6px; padding-left:18px;">
//       <li style="margin-bottom:6px;">
//         <b>Accuracy</b>:
//         predicts the <b>majority class</b> → baseline = majority-class percent.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>Balanced Accuracy</b>:
//         for a predictor that assigns the <b>same class to every row</b> → baseline = <b>1 / #classes</b>.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>Log Loss</b>:
//         predicts the <b>empirical class distribution</b> → baseline = log loss of those class probabilities.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>ROC_AUC</b>:
//         uses a <b>random ranking</b> baseline → baseline = <b>0.50</b>.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>MAE</b>:
//         uses the better of the <b>mean or median</b> constant predictor, usually the median → baseline = the resulting MAE.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>MSE / RMSE</b>:
//         uses the <b>mean of y</b> as the constant prediction → baseline = the resulting MSE or RMSE.
//       </li>
//       <li style="margin-bottom:6px;">
//         <b>RMSLE</b>:
//         clips negatives to 0, averages y in <b>log1p space</b>, converts back to a constant prediction, and reports the resulting RMSLE.
//       </li>
//       <li>
//         <b>R2</b>:
//         the constant-mean predictor corresponds to <b>R2 = 0</b>.
//       </li>
//     </ul>
//   </div>
// `,


// baselineAbsolute: `
//   <div><b>Absolute improvement over the Baseline.</b></div>
//   <div style="margin-top:10px;">
//     This number indicates how much better or worse the model metric is compared to the baseline metric.
//     It is expressed in the same units as the metric itself.
//   </div>
// `,


// metric: `
//   <div><b>The metric that Predictly optimizes.</b></div>

//   <div style="margin-top:8px;">
//     <b>Accuracy:</b> Measures the fraction of correct predictions.
//     Best for balanced classes where all errors matter equally.
//     Range: 0.0-1.0. Higher is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>Balanced_Accuracy:</b> Measures how accurately each class is predicted, giving equal weight to every class.
//     Best for imbalanced datasets where rare classes matter.
//     Range: 0.0-1.0. Higher is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>Log_Loss:</b> Measures how closely predicted probabilities match actual outcomes.
//     Best when prediction confidence matters, not just correctness.
//     Typical values range from near 0.0 upward; lower is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>MAE:</b> Measures the mean absolute error (same units as target).
//     Best for a simple, easy-to-interpret error magnitude.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>MSE:</b> Measures the mean squared error (squared units).
//     Best when large errors should be penalized more heavily.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>R2 (R-squared):</b> Measures the proportion of variance explained relative to predicting the mean.
//     Best for overall regression goodness-of-fit.
//     Typically 0.0-1.0 (can be negative). Higher is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>ROC_AUC:</b> Measures how well the model ranks positive examples above negative ones.
//     In simple terms, it asks: <i>"If we pick a random positive case and a random negative case,
//     how often does the model give the positive one a higher score?"</i>
//     A value of <b>0.50</b> means random guessing, while <b>1.00</b> means perfect ranking.
//     Range: 0.5-1.0. Higher is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>RMSE:</b> Measures the root mean squared error (same units as target).
//     Best when large errors should be penalized more heavily.
//     0.0 is perfect. Lower is better.
//   </div>

//   <div style="margin-top:8px;">
//     <b>RMSLE:</b> Measures the root mean squared logarithmic error.
//     Best when relative (percentage-like) differences matter.
//     0.0 is perfect. Lower is better.
//   </div>
// `,


// ----------------------
// Phase 2 async training
// ----------------------

// async function fetchTrainStatus(userIdForCall: string): Promise<TrainStatusPayload | null> {
//   const resp = await fetchJsonWithTimeout(
//     apiUrl(`/ui/train_status/${encodeURIComponent(userIdForCall)}`),
//     { method: "GET" },
//     TRAIN_STATUS_TIMEOUT_MS,
//     theme
//   );
//   setLastBackendResponse(resp.payload);

//   const norm = normalizeBackendPayload(resp.payload, theme);
//   if (!norm.resultObj) return null;

//   const stateRaw = norm.resultObj.state;
//   const state: TrainStatusState =
//     typeof stateRaw === "string" ? (stateRaw as TrainStatusState) : "UNKNOWN";

//   const updated = norm.resultObj.updated_at_epoch;
//   // Treat 0 / non-number as "no timestamp" instead of "stale"
//   const updated_at_epoch =
//     typeof updated === "number" && updated > 0 ? updated : undefined;

//   // Allow for longer runs: e.g. up to 12 minutes before we call it "stuck".
//   // Note that this is only for long-running jobs, where the status is legitimately "RUNNING" for 12 minutes.
//   // If there is an Exception, then the user should see it almost immediately.
//   const STALE_SECONDS = 12 * 60;
//   const nowEpoch = Math.floor(Date.now() / 1000);

//   // Only call it "stuck" if:
//   //  - we actually have a timestamp, and
//   //  - the state is QUEUED/RUNNING (in-flight), and
//   //  - it has been quiet for longer than STALE_SECONDS.
//   if (
//     updated_at_epoch !== undefined &&
//     (state === "QUEUED" || state === "RUNNING") &&
//     nowEpoch - updated_at_epoch > STALE_SECONDS
//   ) {
//     return {
//       status: "error",
//       state: "FAILED",
//       updated_at_epoch,
//       message:
//         "Predicting appears to be stuck. Please refresh the page or open a new browser window.",
//       error_type: "stale_train_status"
//     };
//   }

//   const msg =
//     typeof norm.resultObj.message === "string" ? norm.resultObj.message : "";
//   const errType =
//     typeof norm.resultObj.error_type === "string" ? norm.resultObj.error_type : "";

//   return {
//     status:
//       typeof norm.resultObj.status === "string" ? norm.resultObj.status : "",
//     state,
//     updated_at_epoch,
//     message: msg,
//     error_type: errType
//   };
// }
