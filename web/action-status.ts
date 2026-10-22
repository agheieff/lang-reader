/** A short status line that clears itself, used for results of actions outside the reader. */

import { byId } from "./dom.js";

const actionStatusMessage = byId("action-status-message");

let actionStatusTimer: number | null = null;

export function showActionStatus(message: string): void {
  if (actionStatusTimer !== null) window.clearTimeout(actionStatusTimer);
  actionStatusMessage.textContent = message;
  actionStatusMessage.hidden = false;
  actionStatusTimer = window.setTimeout(() => {
    actionStatusTimer = null;
    actionStatusMessage.hidden = true;
  }, 5_000);
}
