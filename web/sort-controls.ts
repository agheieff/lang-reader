/** Sort selector and column-header sort buttons shared by the word and character tables. */

import { byId } from "./dom.js";

export interface SortControlsConfig<S extends string> {
  selectId: string;
  buttonSelector: string;
  datasetKey: "wordSort" | "characterSort";
  sorts: readonly S[];
}

export function updateSortControls<S extends string>(
  controls: SortControlsConfig<S>,
  state: { sort: S; direction: "ascending" | "descending" },
  selection: string,
): void {
  byId<HTMLSelectElement>(controls.selectId).value = selection;
  for (const button of document.querySelectorAll<HTMLButtonElement>(controls.buttonSelector)) {
    const sortValue = button.dataset[controls.datasetKey];
    if (!sortValue || !controls.sorts.includes(sortValue as S)) continue;
    const header = button.closest("th");
    if (!header) continue;
    const label = button.dataset.sortLabel ?? button.textContent?.trim() ?? "Column";
    const indicator = button.querySelector<HTMLElement>(".words-sort-indicator");
    const active = state.sort === sortValue;
    if (!active) {
      header.removeAttribute("aria-sort");
      button.setAttribute("aria-label", `Sort by ${label}`);
      button.title = `Sort by ${label}`;
      if (indicator) indicator.textContent = "↕";
      continue;
    }
    header.setAttribute("aria-sort", state.direction);
    const nextDirection = state.direction === "ascending" ? "descending" : "ascending";
    button.setAttribute("aria-label", `${label}, sorted ${state.direction}. Sort ${nextDirection}`);
    button.title = `Sort ${label} ${nextDirection}`;
    if (indicator) indicator.textContent = state.direction === "ascending" ? "↑" : "↓";
  }
}

export function attachSortHeaderButtons<S extends string>(
  controls: SortControlsConfig<S>,
  applySort: (sort: S) => void,
): void {
  for (const button of document.querySelectorAll<HTMLButtonElement>(controls.buttonSelector)) {
    button.addEventListener("click", () => {
      const sortValue = button.dataset[controls.datasetKey];
      if (!sortValue || !controls.sorts.includes(sortValue as S)) return;
      applySort(sortValue as S);
    });
  }
}
