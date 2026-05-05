// src/theme/usePrefersDark.ts

import { useEffect, useState } from "react";

export function usePrefersDark(): boolean {
  const [isDark, setIsDark] = useState<boolean>(() => {
    const mq = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
    return mq?.matches ?? false;
  });

  useEffect(() => {
    const mq = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;

    const handler = (): void => setIsDark(mq.matches);

    // Modern
    mq.addEventListener?.("change", handler);
    // Legacy Safari fallback
    // eslint-disable-next-line deprecation/deprecation
    mq.addListener?.(handler);

    return () => {
      mq.removeEventListener?.("change", handler);
      // eslint-disable-next-line deprecation/deprecation
      mq.removeListener?.(handler);
    };
  }, []);

  return isDark;
}
