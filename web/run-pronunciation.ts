import type { Run } from "./contracts.js";

export function pronunciationForRun(run: Run): string | undefined {
  return run.pronunciation ?? run.term?.pronunciation;
}
