import { describe, expect, it } from "vitest";

import { formatLibraryDate, formatLibraryDateTime } from "./library-date.js";

describe("text library timestamps", () => {
  it("keeps ordinary history dates compact", () => {
    expect(formatLibraryDate("2026-07-10T12:03:00Z", "en-GB", "UTC")).toBe("10 Jul 2026");
  });

  it("shows both the date and time for active preparation", () => {
    expect(formatLibraryDateTime("2026-07-10T12:03:00Z", "en-GB", "UTC")).toBe(
      "10 Jul 2026, 12:03",
    );
  });
});
