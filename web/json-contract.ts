type JsonObject = Record<string, unknown>;

interface JsonContractOptions {
  allowBlankText?: boolean;
  textRequirement?: string;
  enumRequirement?: string;
  timestampRequirement?: string;
}

export function createJsonContract(prefix: string, options: JsonContractOptions = {}) {
  const allowBlankText = options.allowBlankText ?? false;
  const textRequirement = options.textRequirement ?? "must be non-blank text";
  const enumRequirement = options.enumRequirement ?? "is not supported";
  const timestampRequirement = options.timestampRequirement ?? "must be a timestamp";

  function invalid(name: string, requirement: string): never {
    throw new Error(`${prefix}: ${name} ${requirement}`);
  }

  function record(value: unknown, name: string): JsonObject {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return invalid(name, "must be an object");
    }
    return value as JsonObject;
  }

  function text(value: unknown, name: string): string {
    if (typeof value !== "string" || (!allowBlankText && !value.trim())) {
      return invalid(name, textRequirement);
    }
    return value;
  }

  function optionalText(value: unknown, name: string): string | undefined {
    return value === null || value === undefined ? undefined : text(value, name);
  }

  function nullableText(value: unknown, name: string): string | null {
    return value === null ? null : text(value, name);
  }

  function nullishText(value: unknown, name: string): string | null {
    return value === null || value === undefined ? null : text(value, name);
  }

  function integer(value: unknown, name: string, minimum = 0): number {
    if (!Number.isInteger(value) || (value as number) < minimum) {
      return invalid(name, `must be an integer of at least ${minimum}`);
    }
    return value as number;
  }

  function nullableInteger(value: unknown, name: string, minimum = 0): number | null {
    return value === null ? null : integer(value, name, minimum);
  }

  function nonNegativeNumber(value: unknown, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      return invalid(name, "must be a non-negative number");
    }
    return value;
  }

  function boundedNumber(value: unknown, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
      return invalid(name, "must be between 0 and 1");
    }
    return value;
  }

  function number(value: unknown, name: string, minimum = 0, maximum = Infinity): number {
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < minimum ||
      value > maximum
    ) {
      return invalid(name, `must be between ${minimum} and ${maximum}`);
    }
    return value;
  }

  function nullableNumber(
    value: unknown,
    name: string,
    minimum = 0,
    maximum = Infinity,
  ): number | null {
    return value === null ? null : number(value, name, minimum, maximum);
  }

  function boolean(value: unknown, name: string): boolean {
    return typeof value === "boolean" ? value : invalid(name, "must be a boolean");
  }

  function timestamp(value: unknown, name: string, optional?: false): string;
  function timestamp(value: unknown, name: string, optional: true): string | undefined;
  function timestamp(value: unknown, name: string, optional = false): string | undefined {
    if (optional && (value === null || value === undefined)) return undefined;
    const result = text(value, name);
    if (!Number.isFinite(Date.parse(result))) return invalid(name, timestampRequirement);
    return result;
  }

  function oneOf<T extends string>(value: unknown, allowed: readonly T[], name: string): T {
    if (typeof value !== "string" || !allowed.includes(value as T)) {
      return invalid(name, enumRequirement);
    }
    return value as T;
  }

  return {
    record,
    text,
    optionalText,
    nullableText,
    nullishText,
    integer,
    nullableInteger,
    nonNegativeNumber,
    boundedNumber,
    number,
    nullableNumber,
    boolean,
    timestamp,
    oneOf,
  };
}
