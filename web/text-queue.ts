export type TextQueueMoveDirection = "up" | "down";

interface QueueText {
  id: number;
  status: string;
  queue_position: number | null;
}

export function orderedReadyTexts<T extends QueueText>(texts: readonly T[]): T[] {
  return texts
    .filter((item) => item.status === "queued")
    .sort((left, right) => {
      const leftPosition = left.queue_position ?? Number.MAX_SAFE_INTEGER;
      const rightPosition = right.queue_position ?? Number.MAX_SAFE_INTEGER;
      return leftPosition - rightPosition || left.id - right.id;
    });
}

export function queuePositionLabel(position: number): string {
  if (!Number.isInteger(position) || position < 1) {
    throw new Error("Queue position must be a positive integer");
  }
  return String(position);
}

export function readyQueuePositionLabel(text: QueueText): string | null {
  if (text.status !== "queued" || text.queue_position === null) return null;
  return queuePositionLabel(text.queue_position);
}

export function queueNeighborLessonId(
  readyTexts: readonly QueueText[],
  lessonId: number,
  direction: TextQueueMoveDirection,
): number | null {
  const index = readyTexts.findIndex((item) => item.id === lessonId);
  if (index < 0) throw new Error(`Lesson ${lessonId} is not in the ready queue`);
  const neighbor = readyTexts[direction === "up" ? index - 1 : index + 1];
  return neighbor?.id ?? null;
}

export function textQueueMovePath(profileApi: string, lessonId: number): string {
  if (!Number.isInteger(lessonId) || lessonId < 1) throw new Error("Lesson ID must be positive");
  return `${profileApi}/texts/${lessonId}/move`;
}

export function textQueueMovePayload(
  actionId: string,
  direction: TextQueueMoveDirection,
  neighborLessonId: number,
): {
  action_id: string;
  direction: TextQueueMoveDirection;
  neighbor_lesson_id: number;
} {
  if (!actionId.trim()) throw new Error("Action ID must not be blank");
  if (!Number.isInteger(neighborLessonId) || neighborLessonId < 1) {
    throw new Error("Neighbor lesson ID must be positive");
  }
  return {
    action_id: actionId,
    direction,
    neighbor_lesson_id: neighborLessonId,
  };
}
