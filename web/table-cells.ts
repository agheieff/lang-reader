/** Table cells and meters shared by the word and character tables. */

import { textElement } from "./dom.js";

export function wordCell(label: string): HTMLTableCellElement {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  return cell;
}

export function metricCell(label: string, value: number, detail: string): HTMLTableCellElement {
  const cell = wordCell(label);
  cell.append(
    textElement("p", "word-count", value.toLocaleString()),
    textElement("p", "word-count-detail", detail),
  );
  return cell;
}

export function masteryMeter(percent: number, ariaLabel: string, extraClass?: string): HTMLElement {
  const meter = document.createElement("div");
  meter.className = extraClass ? `word-mastery-meter ${extraClass}` : "word-mastery-meter";
  meter.setAttribute("role", "progressbar");
  meter.setAttribute("aria-label", ariaLabel);
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-valuenow", String(percent));
  const fill = document.createElement("div");
  fill.className = "word-mastery-fill";
  fill.style.width = `${percent}%`;
  meter.append(fill);
  return meter;
}

export function formatEvidence(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0+$/, "");
}

export function formatStability(days: number): string {
  if (days < 1) return `${Math.max(1, Math.round(days * 24))}h stability`;
  if (days < 10) return `${days.toFixed(1).replace(/\.0$/, "")}d stability`;
  return `${Math.round(days)}d stability`;
}
