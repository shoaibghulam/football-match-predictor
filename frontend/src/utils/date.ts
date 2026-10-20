/** Date helpers. All calendar dates are local-time "yyyy-mm-dd" strings (the API date format). */

const ISO_DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/
const MS_PER_DAY = 24 * 60 * 60 * 1000

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

/** Local calendar date of `d` as yyyy-mm-dd. */
export function toISODate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

/** Parses yyyy-mm-dd into a local Date at midnight, or null if invalid. */
export function parseISODate(value: string | null | undefined): Date | null {
  if (!value) return null
  const m = ISO_DATE_RE.exec(value)
  if (!m) return null
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])]
  const date = new Date(y, mo - 1, d)
  if (date.getFullYear() !== y || date.getMonth() !== mo - 1 || date.getDate() !== d) return null
  return date
}

export function isValidISODate(value: string | null | undefined): value is string {
  return parseISODate(value) !== null
}

export function todayISO(now: Date = new Date()): string {
  return toISODate(now)
}

/** Adds whole calendar days (DST-safe because it works on local Y/M/D). */
export function addDays(iso: string, days: number): string {
  const d = parseISODate(iso)
  if (!d) throw new Error(`Invalid date: ${iso}`)
  return toISODate(new Date(d.getFullYear(), d.getMonth(), d.getDate() + days))
}

/** Whole days from a to b (b - a). */
export function daysBetween(a: string, b: string): number {
  const da = parseISODate(a)
  const db = parseISODate(b)
  if (!da || !db) throw new Error(`Invalid date range: ${a}..${b}`)
  const utcA = Date.UTC(da.getFullYear(), da.getMonth(), da.getDate())
  const utcB = Date.UTC(db.getFullYear(), db.getMonth(), db.getDate())
  return Math.round((utcB - utcA) / MS_PER_DAY)
}

/** "Today", "Tomorrow", "Yesterday" or e.g. "Thu 23 Oct". */
export function formatDayLabel(iso: string, now: Date = new Date()): string {
  const d = parseISODate(iso)
  if (!d) return iso
  const diff = daysBetween(todayISO(now), iso)
  if (diff === 0) return 'Today'
  if (diff === 1) return 'Tomorrow'
  if (diff === -1) return 'Yesterday'
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })
}

/** Long, unambiguous date such as "Tuesday, 20 October 2026". */
export function formatLongDate(iso: string): string {
  const d = parseISODate(iso)
  if (!d) return iso
  return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
}

export interface Kickoff {
  /** Local calendar date, yyyy-mm-dd, or null when unparseable */
  date: string | null
  /** Local "HH:MM", or null when the kickoff time is unknown */
  time: string | null
}

/** Parses an API kickoff (ISO datetime, or a date-only string when the time is unknown). */
export function parseKickoff(kickoff: string | null | undefined): Kickoff {
  if (!kickoff) return { date: null, time: null }
  if (isValidISODate(kickoff)) return { date: kickoff, time: null }
  const d = new Date(kickoff)
  if (Number.isNaN(d.getTime())) return { date: null, time: null }
  return { date: toISODate(d), time: `${pad(d.getHours())}:${pad(d.getMinutes())}` }
}

/** Anything with an API kickoff, plus the published (UK) date used when the time is unknown. */
export interface Dated {
  kickoff?: string | null
  date?: string | null
}

/** Calendar date of the match on the viewer's clock (falls back to the published date). */
export function localMatchDate(match: Dated): string | null {
  return parseKickoff(match.kickoff).date ?? (isValidISODate(match.date) ? match.date : null)
}

/** True when the match's local date is within [from, to] (inclusive; to defaults to from). */
export function inLocalRange(match: Dated, from: string, to: string = from): boolean {
  const day = localMatchDate(match)
  return day !== null && day >= from && day <= to
}

/** Viewer timezone, e.g. "Asia/Karachi (GMT+5)". */
export function localTimeZoneLabel(now: Date = new Date()): string {
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone
  const offset = new Intl.DateTimeFormat('en-GB', { timeZone: zone, timeZoneName: 'shortOffset' })
    .formatToParts(now)
    .find((p) => p.type === 'timeZoneName')?.value
  return offset ? `${zone} (${offset})` : zone
}

/** Sort key for kickoff strings; unknown kickoffs sort last. */
export function kickoffSortValue(kickoff: string | null | undefined): number {
  if (!kickoff) return Number.MAX_SAFE_INTEGER
  const date = parseISODate(kickoff)
  if (date) return date.getTime()
  const t = new Date(kickoff).getTime()
  return Number.isNaN(t) ? Number.MAX_SAFE_INTEGER : t
}
