export const UNIT_MULTIPLIERS: Record<string, number> = {
  K: 1_000,
  M: 1_000_000,
  B: 1_000_000_000,
  T: 1_000_000_000_000
};

/**
 * Python's `f"{value:.{digits}f}"`.
 *
 * `Number.prototype.toFixed` rounds halves away from zero; Python rounds half to even.
 * That is not a theoretical difference here — a LoRA adapter at 12.5% of base gives
 * exactly 0.125, where Python prints "0.12" and toFixed prints "0.13".
 */
export function toFixedHalfEven(value: number, digits: number): string {
  if (!Number.isFinite(value)) return String(value);

  // Python keeps the sign even when the value rounds to zero: -0.001 formats as "-0.00",
  // and so does -0.0.
  const negative = value < 0 || Object.is(value, -0);
  const abs = Math.abs(value);

  // toFixed switches to exponential notation at 1e21, which would break the digit
  // surgery below. Nothing in this app formats a number that large.
  if (abs >= 1e21) return value.toFixed(digits);

  // Round on the decimal expansion rather than on `abs * 10 ** digits`: scaling is
  // itself lossy and destroys exactly the information this function needs. 0.05 is
  // really 0.05000000000000000277, so Python rounds it up — but 0.05 * 10 collapses to
  // exactly 0.5, which reads as a tie. toFixed is correctly rounded from the binary
  // value, so 25 extra digits is far more than enough to tell a true tie from a near one.
  const [intPart, fracPart = ""] = abs.toFixed(Math.min(digits + 25, 100)).split(".");
  const keep = fracPart.slice(0, digits);
  const rest = fracPart.slice(digits);

  const scaledDigits = intPart + keep;
  let roundUp = false;
  if (rest) {
    const first = rest[0];
    if (first > "5") {
      roundUp = true;
    } else if (first === "5") {
      roundUp = /[1-9]/.test(rest.slice(1))
        ? true
        : (scaledDigits.charCodeAt(scaledDigits.length - 1) - 48) % 2 === 1;
    }
  }

  // BigInt so the carry can't lose precision on large magnitudes.
  const n = BigInt(scaledDigits) + (roundUp ? 1n : 0n);
  const out = n.toString().padStart(digits + 1, "0");
  const result = digits > 0 ? `${out.slice(0, -digits)}.${out.slice(-digits)}` : out;
  return negative ? `-${result}` : result;
}

/** Python's `f"{value:.{precision}e}"` — JS drops the leading zero on 1-digit exponents. */
export function formatE(value: number, precision: number): string {
  return value.toExponential(precision).replace(/e([+-])(\d)$/, "e$10$2");
}

/**
 * Python's `f"{value:.{precision}g}"`.
 *
 * Scientific when the exponent is < -4 or >= precision, fixed otherwise, and trailing
 * zeros are stripped in both cases.
 */
export function formatG(value: number, precision: number): string {
  if (value === 0) return "0";
  if (!Number.isFinite(value)) return String(value);

  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const strip = (s: string) => (s.includes(".") ? s.replace(/\.?0+$/, "") : s);

  if (exponent < -4 || exponent >= precision) {
    const [mantissa, exp] = formatE(value, precision - 1).split("e");
    return `${strip(mantissa)}e${exp}`;
  }
  return strip(toFixedHalfEven(value, Math.max(0, precision - 1 - exponent)));
}

export function money(n: number, cents = false): string {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    ...(cents ? { minimumFractionDigits: 2 } : { maximumFractionDigits: 0 })
  });
}

export function fmtTokens(n: number): string {
  if (n >= 1_000_000_000_000) return `${toFixedHalfEven(n / 1_000_000_000_000, 2)}T`;
  if (n >= 1_000_000_000) return `${toFixedHalfEven(n / 1_000_000_000, 2)}B`;
  if (n >= 1_000_000) return `${toFixedHalfEven(n / 1_000_000, 2)}M`;
  if (n >= 1_000) return `${toFixedHalfEven(n / 1_000, 1)}K`;
  return String(n);
}

export function fmtSamples(n: number): string {
  if (n >= 1_000_000_000) return `${toFixedHalfEven(n / 1_000_000_000, 2)}B`;
  if (n >= 1_000_000) return `${toFixedHalfEven(n / 1_000_000, 2)}M`;
  if (n >= 1_000) return `${toFixedHalfEven(n / 1_000, 1)}K`;
  return String(n);
}

export function formatWallClockTime(wallClockDays: number): string {
  const wallClockHours = wallClockDays * 24;
  if (wallClockDays >= 1) return `${toFixedHalfEven(wallClockDays, 2)} days`;
  if (wallClockDays >= 1 / 24) return `${toFixedHalfEven(wallClockHours, 2)} hrs`;
  return `${toFixedHalfEven(wallClockHours * 60, 1)} min`;
}

export function formatGpuHours(gpuHours: number): string {
  if (gpuHours >= 1_000_000) return `${toFixedHalfEven(gpuHours / 1_000_000, 2)}M`;
  if (gpuHours >= 1_000) return `${toFixedHalfEven(gpuHours / 1_000, 1)}K`;
  return toFixedHalfEven(gpuHours, 1);
}

/**
 * Format a value for the summary table.
 *
 * `isFloat` distinguishes Python's float from int, which JS cannot: Python's 1000.0 and
 * 1000 format differently ("1.0K" vs "1,000"). Callers holding a genuinely fractional
 * quantity should pass true.
 */
export function displayValue(val: number | boolean | string, isFloat = false): string {
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (typeof val === "string") return val;

  if (isFloat || !Number.isInteger(val)) {
    if (val >= 1e12) return formatE(val, 2);
    if (val >= 1e9) return `${toFixedHalfEven(val / 1e9, 2)}B`;
    if (val >= 1e6) return `${toFixedHalfEven(val / 1e6, 2)}M`;
    if (val >= 1e3) return `${toFixedHalfEven(val / 1e3, 1)}K`;
    return formatG(val, 4);
  }
  if (val >= 1_000) return val.toLocaleString("en-US");
  return String(val);
}
