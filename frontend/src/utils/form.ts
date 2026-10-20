import type { FormResult, TeamForm } from '../api/types'

export type ResultLetter = 'W' | 'D' | 'L'

export interface FormEntry {
  key: string
  outcome: ResultLetter | null
  opponent: string
  venue: 'H' | 'A' | null
  scoreText: string | null
  date: string | null
}

export interface RatingEntry {
  key: string
  label: string
  value: string
}

const RATING_LABELS: Record<string, string> = {
  elo: 'Elo rating',
  pi_home: 'Pi-rating (home)',
  pi_away: 'Pi-rating (away)',
  form5: 'Form (last 5)',
}

function letterFromGoals(gf: number, ga: number): ResultLetter {
  if (gf > ga) return 'W'
  if (gf < ga) return 'L'
  return 'D'
}

function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

function normalizeLetter(r: string | undefined): ResultLetter | null {
  const c = r?.trim().charAt(0).toUpperCase()
  return c === 'W' || c === 'D' || c === 'L' ? c : null
}

/** Normalises one past result to the given team's perspective. */
export function toFormEntry(r: FormResult, team: string, index: number): FormEntry {
  const t = team.toLowerCase()
  const isHome = r.home ? r.home.toLowerCase() === t : r.venue?.toUpperCase().startsWith('H') ?? null
  const venue = isHome === null ? null : isHome ? 'H' : 'A'
  const opponent = r.opponent ?? (isHome ? r.away : r.home) ?? 'Unknown'

  let gf: number | null = isNum(r.goals_for) ? r.goals_for : null
  let ga: number | null = isNum(r.goals_against) ? r.goals_against : null
  if ((gf === null || ga === null) && isNum(r.home_goals) && isNum(r.away_goals) && isHome !== null) {
    gf = isHome ? r.home_goals : r.away_goals
    ga = isHome ? r.away_goals : r.home_goals
  }

  const outcome = normalizeLetter(r.result) ?? (gf !== null && ga !== null ? letterFromGoals(gf, ga) : null)
  return {
    key: `${r.date ?? index}-${opponent}-${index}`,
    outcome,
    opponent,
    venue,
    scoreText: gf !== null && ga !== null ? `${gf}-${ga}` : null,
    date: r.date ?? null,
  }
}

function pickResults(form: TeamForm): FormResult[] {
  const list = form.results ?? form.matches ?? form.last_results
  return Array.isArray(list) ? list : []
}

/** Most recent first, as returned by the API (max 10). */
export function normalizeForm(form: TeamForm | undefined, team: string): FormEntry[] {
  if (!form) return []
  return pickResults(form)
    .slice(0, 10)
    .map((r, i) => toFormEntry(r, team, i))
}

function humanize(key: string): string {
  const s = key.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function normalizeRatings(form: TeamForm | undefined): RatingEntry[] {
  const ratings = form?.ratings
  if (!ratings || typeof ratings !== 'object') return []
  return Object.entries(ratings)
    .filter((e): e is [string, number] => isNum(e[1]))
    .map(([key, v]) => ({
      key,
      label: RATING_LABELS[key] ?? humanize(key),
      value: key === 'elo' ? String(Math.round(v)) : v.toFixed(2),
    }))
}

export function summarizeForm(entries: readonly FormEntry[]): Record<ResultLetter, number> {
  return entries.reduce(
    (acc, e) => (e.outcome ? { ...acc, [e.outcome]: acc[e.outcome] + 1 } : acc),
    { W: 0, D: 0, L: 0 } as Record<ResultLetter, number>,
  )
}
