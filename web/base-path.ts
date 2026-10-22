// The server renders its public prefix (empty locally, e.g. `/lang` behind a reverse proxy) on <body>.
export function basePath(): string {
  if (typeof document === "undefined") return "";
  return document.body?.dataset.basePath ?? "";
}

export function appPath(path: string, base: string = basePath()): string {
  if (!path.startsWith("/")) throw new Error("Application paths must be absolute");
  return `${base}${path}`;
}
