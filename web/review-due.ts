export function isReviewDue(entry: { next_due_at?: string | undefined }, now: Date): boolean {
  return Boolean(entry.next_due_at && new Date(entry.next_due_at).getTime() <= now.getTime());
}

export function formatReviewDue(nextDueAt: string | undefined, now = new Date()): string {
  if (!nextDueAt) return "Not scheduled";
  const milliseconds = new Date(nextDueAt).getTime() - now.getTime();
  if (milliseconds <= 0) return "Due now";
  const days = Math.ceil(milliseconds / 86_400_000);
  if (days === 1) return "Due within a day";
  return `Due in ${days} days`;
}
