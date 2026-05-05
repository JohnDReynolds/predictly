// src/api/uiClient.ts

import type { Theme } from "../theme/theme";

/**
 * Internal: resolve API base URL from VITE_API_BASE_URL.
 * Trailing slashes are trimmed.
 */
function getApiBaseUrl(): string {
  const v = String(import.meta.env.VITE_API_BASE_URL ?? "").trim();
  return v.replace(/\/+$/, "");
}

/**
 * Build an absolute API URL, or fall back to the path if no base is set.
 */
export function apiUrl(path: string): string {
  const base = getApiBaseUrl();
  if (!base) return path;
  return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

/**
 * Fetch JSON with a timeout and normalized error payload.
 *
 * NOTE: `theme` is currently unused here, but kept in the signature
 * to avoid touching existing call sites in App.tsx.
 */
export async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  theme: Theme
): Promise<{ ok: boolean; httpStatus: number; payload: unknown }> {
  const controller = new AbortController();
  const t = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    const httpStatus = res.status;

    const ct = res.headers.get("content-type") ?? "";
    if (!ct.toLowerCase().includes("application/json")) {
      const payload = {
        result: {
          status: "error",
          error_type: "non_json_response",
          message: `Fetch request failed (HTTP ${httpStatus}): expected JSON.  Please refresh the page or open a new browser window.`,
          http_status: httpStatus,
          content_type: ct || "",
          url
        }
      };
      return { ok: false, httpStatus, payload };
    }

    const data = (await res.json()) as unknown;
    return { ok: res.ok, httpStatus, payload: data };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      const payload = {
        result: {
          status: "error",
          error_type: "timeout",
          message: `Fetch request timed out after ${Math.round(timeoutMs / 1000)}s.  Please refresh the page or open a new browser window.`,
          url
        }
      };
      return { ok: false, httpStatus: 0, payload };
    }

    const message = err instanceof Error ? err.message : String(err);
    const payload = {
      result: {
        status: "error",
        error_type: "network_error",
        message: `Network error: ${message}.  Please refresh the page or open a new browser window.`,
        url
      }
    };
    return { ok: false, httpStatus: 0, payload };
  } finally {
    globalThis.clearTimeout(t);
  }
}
