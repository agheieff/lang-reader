import { describe, expect, it } from "vitest";

import { isDialogueSentence } from "./dialogue-format.js";

describe("dialogue sentence formatting", () => {
  it.each([
    "陈雨问：“你听清最后一句了吗？”",
    "“我改变主意了，”他说，“坐火车更慢。”",
    "—Deberíamos tomar el autobús —dijo Ana.",
    "„Wir warten noch zehn Minuten“, sagte Mara.",
    '"We can leave now," Alex said.',
  ])("recognizes a direct-speech turn: %s", (text) => {
    expect(isDialogueSentence(text)).toBe(true);
  });

  it.each([
    "陈雨和高明还在车站里等车。",
    "The sign used the term “quiet zone” for this carriage.",
    "Ein sogenannter „Ruhebereich“ liegt im zweiten Wagen.",
    "La palabra «andén» aparece en la pantalla.",
  ])("leaves ordinary prose inline: %s", (text) => {
    expect(isDialogueSentence(text)).toBe(false);
  });
});
