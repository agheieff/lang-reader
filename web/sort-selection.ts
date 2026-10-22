export type SortDirection = "ascending" | "descending";

export interface SortSelectionState<S extends string> {
  sort: S;
  direction: SortDirection;
}

export interface SortSelection<S extends string> {
  defaultState: SortSelectionState<S>;
  next(current: SortSelectionState<S>, sort: S): SortSelectionState<S>;
  parse(value: string): SortSelectionState<S>;
  selection(state: SortSelectionState<S>): string;
  reverseResult(state: SortSelectionState<S>): boolean;
}

/**
 * Shared table-sort state machine: header clicks pick a column with its default
 * direction and toggle on repeat, dropdown values round-trip as "sort:direction",
 * and `reverseResult` reports when the comparator's native order must be reversed.
 */
export function createSortSelection<S extends string>(
  sorts: readonly S[],
  defaultSort: S,
  defaultAscendingSorts: readonly S[],
  nativeAscendingSorts: readonly S[],
): SortSelection<S> {
  const defaultDirection = (sort: S): SortDirection =>
    defaultAscendingSorts.includes(sort) ? "ascending" : "descending";
  const nativeDirection = (sort: S): SortDirection =>
    nativeAscendingSorts.includes(sort) ? "ascending" : "descending";
  const defaultState: SortSelectionState<S> = {
    sort: defaultSort,
    direction: defaultDirection(defaultSort),
  };

  return {
    defaultState,
    next(current, sort) {
      if (current.sort !== sort) return { sort, direction: defaultDirection(sort) };
      return {
        sort,
        direction: current.direction === "ascending" ? "descending" : "ascending",
      };
    },
    parse(value) {
      const separator = value.lastIndexOf(":");
      if (separator < 0) return defaultState;
      const sort = value.slice(0, separator);
      const direction = value.slice(separator + 1);
      if (!sorts.includes(sort as S) || (direction !== "ascending" && direction !== "descending")) {
        return defaultState;
      }
      return { sort: sort as S, direction };
    },
    selection(state) {
      return `${state.sort}:${state.direction}`;
    },
    reverseResult(state) {
      return state.direction !== nativeDirection(state.sort);
    },
  };
}
