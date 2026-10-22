/** The first-run questionnaire that activates a language profile. */

import {
  type ProfileActivation,
  type ProfileActivationUpdate,
  parseInterestInput,
  parseProfileActivation,
  QUESTIONNAIRE_CONFIDENCES,
  type QuestionnaireConfidence,
  STARTING_POINTS,
  type StartingPoint,
} from "./activation.js";
import { getJson, postJson, profileApi } from "./api.js";
import { byId } from "./dom.js";

export async function loadActivation(): Promise<ProfileActivation> {
  return parseProfileActivation(await getJson(`${profileApi}/activation`, "Activation status"));
}

async function submitActivation(): Promise<void> {
  const selected = document.querySelector<HTMLInputElement>('input[name="starting-point"]:checked');
  const startingPoint = selected?.value;
  if (!startingPoint || !STARTING_POINTS.includes(startingPoint as StartingPoint)) {
    throw new Error("Choose a reading starting point.");
  }
  const confidenceValue = byId<HTMLSelectElement>("activation-confidence").value;
  if (!QUESTIONNAIRE_CONFIDENCES.includes(confidenceValue as QuestionnaireConfidence)) {
    throw new Error("Choose how confident you are in the estimate.");
  }
  const textLengthValue = byId<HTMLSelectElement>("activation-text-length").value;
  const textLength = textLengthValue ? Number(textLengthValue) : undefined;
  const payload: ProfileActivationUpdate = {
    starting_point: startingPoint as StartingPoint,
    confidence: confidenceValue as QuestionnaireConfidence,
    interests: parseInterestInput(byId<HTMLInputElement>("activation-interests").value),
    ...(textLength === undefined ? {} : { text_length: textLength }),
  };
  const submit = byId<HTMLButtonElement>("activation-submit");
  const status = byId("activation-status");
  submit.disabled = true;
  submit.textContent = "Starting…";
  status.textContent = "Saving your starting point…";
  try {
    const response = await postJson(`${profileApi}/activation`, payload);
    if (!response.ok) throw new Error(`Activation returned ${response.status}`);
    const activated = parseProfileActivation(await response.json());
    if (!activated.active) throw new Error("The language profile was not activated.");
    window.location.reload();
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Activation failed.";
    submit.disabled = false;
    submit.textContent = "Start this language";
  }
}

export function initActivationView(): void {
  byId("activation-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void submitActivation();
  });
}
