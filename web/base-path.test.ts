import { describe, expect, it } from "vitest";

import { appPath } from "./base-path.js";

describe("application base path", () => {
  it("prefixes absolute application paths", () => {
    expect(appPath("/p/zh-hans", "/lang")).toBe("/lang/p/zh-hans");
    expect(appPath("/api/profiles", "")).toBe("/api/profiles");
  });

  it("rejects relative paths", () => {
    expect(() => appPath("p/zh-hans", "/lang")).toThrow();
  });
});
