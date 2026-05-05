// src/utils/uiResults.ts

import type { Theme } from "../theme/theme";

export type UiMessage = { text: string; color: string };

export type NormalizedBackend = {
  resultObj: Record<string, unknown> | null;
  status: string;
  msg: UiMessage | null;
};

function isPlainObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function tryParseJson(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function unwrapUiResult(payload: unknown): Record<string, unknown> | null {
  if (!isPlainObject(payload)) return null;
  const r = payload.result;
  if (!isPlainObject(r)) return null;
  return r;
}

function normalizeResultFields(resultObj: Record<string, unknown>): Record<string, unknown> {
  const responseText = resultObj.response_text;
  if (typeof responseText !== "string" || !responseText.trim()) return resultObj;

  const parsed = tryParseJson(responseText);
  if (!isPlainObject(parsed)) return resultObj;

  const next: Record<string, unknown> = { ...resultObj };
  const et = (parsed as Record<string, unknown>).error_type;
  const msg = (parsed as Record<string, unknown>).message;

  if (typeof et === "string" && et.trim()) next.error_type = et.trim();
  if (typeof msg === "string" && msg.trim()) next.message = msg.trim();

  return next;
}

export function normalizeBackendPayload(payload: unknown, theme: Theme): NormalizedBackend {
  const base = unwrapUiResult(payload);

  if (!base) {
    return {
      resultObj: null,
      status: "",
      msg: { color: theme.danger, text: "bad_response_shape - Response was not understood." }
    };
  }

  const resultObj = normalizeResultFields(base);
  const status = typeof resultObj.status === "string" ? resultObj.status.toLowerCase() : "";

  const message = typeof resultObj.message === "string" ? resultObj.message : "";
  const errorType = typeof resultObj.error_type === "string" ? resultObj.error_type : "";

  const msg: UiMessage | null =
    !message && !errorType
      ? null
      : {
        color: status === "error" ? theme.danger : theme.text2,
        // Do not show error_type to user directly
        // text: errorType ? `${errorType} - ${message}` : message
        text: errorType ? `${message}` : message
      };

  if (status === "error" && !msg) {
    return {
      resultObj,
      status,
      msg: { color: theme.danger, text: "backend_error - Request failed." }
    };
  }

  return { resultObj, status, msg };
}

export function normalizeRecords(val: unknown): Array<Record<string, unknown>> {
  if (!val) return [];
  if (Array.isArray(val)) {
    return val.filter((x) => isPlainObject(x)) as Array<Record<string, unknown>>;
  }
  if (typeof val === "string") {
    const parsed = tryParseJson(val);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x) => isPlainObject(x)) as Array<Record<string, unknown>>;
  }
  return [];
}

function toNumberOrNaN(x: unknown): number {
  return typeof x === "number" ? x : Number.NaN;
}

/**
 * Format a numeric value to 4 decimal places, or fall back to string if not finite.
 */
export function fmt4(x: unknown): string {
  const n = toNumberOrNaN(x);
  if (!Number.isFinite(n)) return String(x ?? "");
  const rounded = Math.round(n * 1e4) / 1e4;
  return rounded.toFixed(4);
}

/**
 * Format a metric value, optionally as a percentage.
 */
export function fmtMetric(x: unknown, percent: boolean = false): string {
  const n = toNumberOrNaN(x);
  if (!Number.isFinite(n)) return String(x ?? "");

  if (percent) {
    const roundedPct = Math.round(n * 100 * 1e2) / 1e2;
    return `${roundedPct.toFixed(2)}%`;
  }

  const rounded = Math.round(n * 1e4) / 1e4;
  return rounded.toFixed(4);
}
