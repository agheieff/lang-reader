import { describe, expect, it } from "vitest";

import {
  orderedReadyTexts,
  queueNeighborLessonId,
  queuePositionLabel,
  readyQueuePositionLabel,
  textQueueMovePath,
  textQueueMovePayload,
} from "./text-queue.js";

const queue = [
  { id: 9, status: "queued", queue_position: 4 },
  { id: 2, status: "in_progress", queue_position: 1 },
  { id: 7, status: "queued", queue_position: 2 },
  { id: 3, status: "read", queue_position: null },
  { id: 8, status: "queued", queue_position: 3 },
];

describe("ready text queue", () => {
  it("orders only ready texts by their durable queue positions", () => {
    expect(orderedReadyTexts(queue).map((item) => item.id)).toEqual([7, 8, 9]);
  });

  it("finds adjacent lessons from the ready subset despite global position gaps", () => {
    const ready = orderedReadyTexts(queue);

    expect(queueNeighborLessonId(ready, 7, "up")).toBeNull();
    expect(queueNeighborLessonId(ready, 7, "down")).toBe(8);
    expect(queueNeighborLessonId(ready, 8, "up")).toBe(7);
    expect(queueNeighborLessonId(ready, 8, "down")).toBe(9);
    expect(queueNeighborLessonId(ready, 9, "down")).toBeNull();
  });

  it("shows queue positions as compact numbers", () => {
    expect(queuePositionLabel(1)).toBe("1");
    expect(queuePositionLabel(2)).toBe("2");
    expect(queuePositionLabel(12)).toBe("12");
    expect(() => queuePositionLabel(0)).toThrow(/positive integer/);
  });

  it("only gives ready cards a position label", () => {
    expect(readyQueuePositionLabel({ id: 1, status: "queued", queue_position: 2 })).toBe("2");
    expect(
      readyQueuePositionLabel({ id: 2, status: "in_progress", queue_position: null }),
    ).toBeNull();
    expect(readyQueuePositionLabel({ id: 3, status: "skipped", queue_position: null })).toBeNull();
    expect(readyQueuePositionLabel({ id: 4, status: "read", queue_position: null })).toBeNull();
  });

  it("fails fast when a rendered lesson is absent from the supplied ready queue", () => {
    expect(() => queueNeighborLessonId(orderedReadyTexts(queue), 99, "up")).toThrow(
      /not in the ready queue/,
    );
  });
});

describe("queue move requests", () => {
  it("builds a profile-scoped move with its expected adjacent lesson", () => {
    expect(textQueueMovePath("/api/profiles/zh-hans", 8)).toBe(
      "/api/profiles/zh-hans/texts/8/move",
    );
    expect(textQueueMovePayload("move-1", "up", 7)).toEqual({
      action_id: "move-1",
      direction: "up",
      neighbor_lesson_id: 7,
    });
  });

  it("rejects invalid action and lesson identities", () => {
    expect(() => textQueueMovePath("/api/profiles/es-es", 0)).toThrow(/positive/);
    expect(() => textQueueMovePayload(" ", "down", 7)).toThrow(/blank/);
    expect(() => textQueueMovePayload("move-1", "down", 0)).toThrow(/positive/);
  });
});
