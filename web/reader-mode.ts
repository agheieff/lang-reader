import { appPath } from "./base-path.js";

export type ReaderMode = "standard" | "reread" | "preview";
export type TextAction = "read" | "continue" | "reread" | "preview";

export interface ReaderLaunch {
  lessonId?: number | undefined;
  mode: ReaderMode;
}

export function readerModeRecordsEvidence(mode: ReaderMode): boolean {
  return mode !== "preview";
}

export function parseReaderLaunch(search: string): ReaderLaunch {
  const parameters = new URLSearchParams(search);
  const rawLesson = parameters.get("lesson_id");
  const rawMode = parameters.get("mode");
  if (rawLesson === null) return { mode: "standard" };
  if (!/^\d+$/.test(rawLesson) || Number(rawLesson) < 1) {
    throw new Error("The selected text ID is invalid.");
  }
  if (rawMode !== null && rawMode !== "preview" && rawMode !== "reread") {
    throw new Error("The selected reading mode is invalid.");
  }
  return {
    lessonId: Number(rawLesson),
    mode: rawMode === "preview" ? "preview" : rawMode === "reread" ? "reread" : "standard",
  };
}

export function textActionHref(profileId: string, lessonId: number, action: TextAction): string {
  if (!Number.isInteger(lessonId) || lessonId < 1) throw new Error("Lesson ID must be positive");
  const parameters = new URLSearchParams({ lesson_id: String(lessonId) });
  if (action === "preview") parameters.set("mode", "preview");
  if (action === "reread") parameters.set("mode", "reread");
  return appPath(`/p/${encodeURIComponent(profileId)}?${parameters.toString()}`);
}

export function readerApiPath(profileApi: string, launch: ReaderLaunch): string {
  if (launch.lessonId === undefined) return `${profileApi}/reader`;
  if (launch.mode === "preview") return `${profileApi}/texts/${launch.lessonId}`;
  const parameters = new URLSearchParams({ lesson_id: String(launch.lessonId) });
  if (launch.mode === "reread") parameters.set("fresh", "true");
  return `${profileApi}/reader?${parameters.toString()}`;
}
