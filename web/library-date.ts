export function formatLibraryDate(
  value: string,
  locales?: Intl.LocalesArgument,
  timeZone?: string,
): string {
  return new Intl.DateTimeFormat(locales, {
    dateStyle: "medium",
    ...(timeZone ? { timeZone } : {}),
  }).format(new Date(value));
}

export function formatLibraryDateTime(
  value: string,
  locales?: Intl.LocalesArgument,
  timeZone?: string,
): string {
  return new Intl.DateTimeFormat(locales, {
    dateStyle: "medium",
    timeStyle: "short",
    ...(timeZone ? { timeZone } : {}),
  }).format(new Date(value));
}
