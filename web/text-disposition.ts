export type TextDisposition = "skip" | "restore";

export function setTextCardActionPending(
  button: HTMLButtonElement,
  pending: boolean,
  label?: string,
  card: HTMLElement | null = button.closest<HTMLElement>(".text-card"),
): HTMLElement | null {
  button.disabled = pending;
  if (label !== undefined) button.textContent = label;
  if (pending) button.setAttribute("aria-busy", "true");
  else button.removeAttribute("aria-busy");
  if (card) {
    card.inert = pending;
    if (pending) card.setAttribute("aria-busy", "true");
    else card.removeAttribute("aria-busy");
  }
  return card;
}

export function textDispositionPath(
  profileApi: string,
  lessonId: number,
  disposition: TextDisposition,
): string {
  if (!Number.isInteger(lessonId) || lessonId < 1) throw new Error("Lesson ID must be positive");
  return `${profileApi}/texts/${lessonId}/${disposition}`;
}

export function textDispositionPayload(actionId: string): { action_id: string } {
  if (!actionId.trim()) throw new Error("Action ID must not be blank");
  return { action_id: actionId };
}
