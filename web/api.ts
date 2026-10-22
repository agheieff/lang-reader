/** The active profile and the JSON helpers every view uses to reach its API. */

import { appPath } from "./base-path.js";

const rawProfileId = document.body.dataset.profileId;
if (!rawProfileId) throw new Error("Missing active profile ID");
export const profileId: string = rawProfileId;
export const profileApi = appPath(`/api/profiles/${encodeURIComponent(profileId)}`);

export async function getJson(
  path: string,
  failure: string | ((status: number) => Error),
): Promise<unknown> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw typeof failure === "string"
      ? new Error(`${failure} returned ${response.status}`)
      : failure(response.status);
  }
  return response.json();
}

export function postJson(path: string, body: unknown, method = "POST"): Promise<Response> {
  return fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}
