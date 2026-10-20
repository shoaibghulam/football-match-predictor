/** Deterministic, realistic sample data for the mock API (dev demo mode and tests). */
import type { Fixture, FormResult, League, Prediction, ScoreProb, Team, TeamForm } from '../api/types'
import { addDays, daysBetween, todayISO } from '../utils/date'

export const MOCK_MODEL_VERSION = '2026-10-20'

export const LEAGUES: League[] = [
  { code: 'UCL', name: 'UEFA Champions League', country: 'Europe' },
  { code: 'UEL', name: 'UEFA Europa League', country: 'Europe' },
  { code: 'UECL', name: 'UEFA Conference League', country: 'Europe' },
  { code: 'E0', name: 'Premier League', country: 'England' },
  { code: 'E1', name: 'Championship', country: 'England' },
  { code: 'SP1', name: 'La Liga', country: 'Spain' },
  { code: 'D1', name: 'Bundesliga', country: 'Germany' },
  { code: 'I1', name: 'Serie A', country: 'Italy' },
]

const TEAM_ELO: Record<string, [string, number][]> = {
  E0: [['Arsenal', 1985], ['Liverpool', 1990], ['Man City', 1970], ['Chelsea', 1890], ['Newcastle', 1860], ['Aston Villa', 1850], ['Brighton', 1800], ['Everton', 1740]],
  E1: [['Leeds', 1720], ['Sunderland', 1700], ['Middlesbrough', 1680], ['Norwich', 1650], ['Coventry', 1660], ['Hull', 1610]],
  SP1: [['Real Madrid', 1995], ['Barcelona', 1980], ['Ath Madrid', 1900], ['Ath Bilbao', 1840], ['Sociedad', 1800], ['Getafe', 1730]],
  D1: [['Bayern Munich', 1990], ['Leverkusen', 1900], ['Dortmund', 1870], ['RB Leipzig', 1850], ['Stuttgart', 1820], ['Mainz', 1740]],
  I1: [['Inter', 1960], ['Napoli', 1900], ['Juventus', 1880], ['Milan', 1870], ['Roma', 1830], ['Bologna', 1810]],
}

export const TEAMS: Team[] = Object.entries(TEAM_ELO).flatMap(([league, list]) =>
  list.map(([name]) => ({ name, league })),
)

const EUROPE_ONLY_ELO: [string, number][] = [['Paris SG', 1920], ['Benfica', 1800], ['Celtic', 1700]]
const ELO = new Map([...Object.values(TEAM_ELO).flatMap((list) => list), ...EUROPE_ONLY_ELO])
const UEFA = new Set(['UCL', 'UEL', 'UECL'])

export function findTeam(name: string): Team | undefined {
  const n = name.trim().toLowerCase()
  return TEAMS.find((t) => t.name.toLowerCase() === n)
}

/** Small deterministic string hash (FNV-1a). */
export function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function poisson(k: number, lambda: number): number {
  let f = 1
  for (let i = 2; i <= k; i += 1) f *= i
  return (Math.exp(-lambda) * lambda ** k) / f
}

const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d
const MAX_GOALS = 7
const HOME_ADVANTAGE_ELO = 65

export function mockPrediction(home: string, away: string, useMarket = true): Prediction {
  const diff = (ELO.get(home) ?? 1750) + HOME_ADVANTAGE_ELO - (ELO.get(away) ?? 1750)
  const lh = 1.38 * Math.exp(diff / 900)
  const la = 1.12 * Math.exp(-diff / 900)
  let ph = 0
  let pd = 0
  let pa = 0
  let over = 0
  let btts = 0
  const scores: ScoreProb[] = []
  for (let h = 0; h <= MAX_GOALS; h += 1) {
    for (let a = 0; a <= MAX_GOALS; a += 1) {
      const p = poisson(h, lh) * poisson(a, la)
      if (h > a) ph += p
      else if (h === a) pd += p
      else pa += p
      if (h + a > 2) over += p
      if (h > 0 && a > 0) btts += p
      scores.push({ score: `${h}-${a}`, p })
    }
  }
  const total = ph + pd + pa
  return {
    probs: { home: round(ph / total), draw: round(pd / total), away: round(pa / total) },
    expected_goals: { home: round(lh, 2), away: round(la, 2) },
    over_2_5: round(over / total),
    btts: round(btts / total),
    top_scores: scores
      .sort((x, y) => y.p - x.p)
      .slice(0, 5)
      .map((s) => ({ score: s.score, p: round(s.p / total) })),
    model: useMarket ? 'market' : 'base',
    model_version: MOCK_MODEL_VERSION,
  }
}

const KICKOFFS = ['12:30', '15:00', '17:30', '20:00']

export function fixturesForDate(date: string, today: string = todayISO()): Fixture[] {
  return LEAGUES.filter((l) => !UEFA.has(l.code)).flatMap((league) => {
    const teams = TEAM_ELO[league.code] ?? []
    const seed = hash(`${date}|${league.code}`)
    const count = seed % 4 // 0-3 matches per league per day
    const order = [...teams].sort((a, b) => hash(`${seed}${a[0]}`) - hash(`${seed}${b[0]}`))
    return Array.from({ length: Math.min(count, Math.floor(order.length / 2)) }, (_, i) => {
      const home = order[i * 2]?.[0] ?? ''
      const away = order[i * 2 + 1]?.[0] ?? ''
      const prediction = mockPrediction(home, away, (seed + i) % 3 !== 0)
      const finished = daysBetween(date, today) > 0
      const top = prediction.top_scores[(seed + i) % prediction.top_scores.length]?.score ?? '1-1'
      const [hg, ag] = top.split('-').map(Number)
      return {
        id: `${date}-${league.code}-${i}`,
        league: league.code,
        kickoff: `${date}T${KICKOFFS[(seed + i) % KICKOFFS.length]}:00`,
        home,
        away,
        odds: null,
        prediction,
        status: finished ? 'finished' : 'scheduled',
        score: finished ? { home: hg ?? 0, away: ag ?? 0 } : null,
      }
    })
  })
}

/** UEFA sample for "today": Barcelona v Paris SG in play plus two evening kick-offs. */
export function uefaFixturesForDate(date: string, today: string = todayISO()): Fixture[] {
  if (date !== today) return []
  const europe = (home: string, away: string) => ({ ...mockPrediction(home, away, false), model: 'europe' })
  return [
    {
      id: `${date}-UCL-live`,
      league: 'UCL',
      kickoff: `${date}T20:00:00`,
      home: 'Barcelona',
      away: 'Paris SG',
      odds: null,
      prediction: europe('Barcelona', 'Paris SG'),
      status: 'live',
      score: { home: 2, away: 1 },
      competition_type: 'europe',
      stage: 'league',
      featured: true,
      live: { minute: 67, minute_estimated: false, score: { home: 2, away: 1 } },
      prediction_frozen_at: `${date}T18:30:00Z`,
    },
    {
      id: `${date}-UCL-1`,
      league: 'UCL',
      kickoff: `${date}T20:00:00`,
      home: 'Bayern Munich',
      away: 'Arsenal',
      odds: null,
      prediction: europe('Bayern Munich', 'Arsenal'),
      status: 'scheduled',
      score: null,
      competition_type: 'europe',
      stage: 'league',
      featured: true,
      live: null,
    },
    {
      id: `${date}-UEL-1`,
      league: 'UEL',
      kickoff: `${date}T17:45:00`,
      home: 'Celtic',
      away: 'Benfica',
      odds: null,
      prediction: { ...europe('Celtic', 'Benfica'), confidence: 'low' },
      status: 'scheduled',
      score: null,
      competition_type: 'europe',
      stage: 'league',
      featured: true,
      live: null,
    },
  ]
}

export function fixturesForRange(from: string, to: string, today?: string): Fixture[] {
  const days = Math.max(0, daysBetween(from, to))
  return Array.from({ length: days + 1 }, (_, i) => {
    const date = addDays(from, i)
    return [...uefaFixturesForDate(date, today), ...fixturesForDate(date, today)]
  }).flat()
}

/** Live first, then UEFA, then the strongest domestic pairings (max 12) — mirrors /featured. */
export function featuredForDate(date: string, today?: string): Fixture[] {
  const all = fixturesForRange(date, date, today)
  const strength = (f: Fixture) => (ELO.get(f.home) ?? 1500) + (ELO.get(f.away) ?? 1500)
  const live = all.filter((f) => f.status === 'live')
  const uefa = all.filter((f) => f.status !== 'live' && UEFA.has(f.league))
  const big = all
    .filter((f) => !UEFA.has(f.league) && f.status !== 'live')
    .filter((f) => Math.min(ELO.get(f.home) ?? 0, ELO.get(f.away) ?? 0) >= 1880)
    .sort((a, b) => strength(b) - strength(a))
  return [...live, ...uefa, ...big].slice(0, 12).map((f) => ({ ...f, featured: true }))
}

export function mockTeamForm(team: Team, today: string = todayISO()): TeamForm {
  const rivals = TEAMS.filter((t) => t.league === team.league && t.name !== team.name)
  const results: FormResult[] = Array.from({ length: 10 }, (_, i) => {
    const rival = rivals[(hash(`${team.name}${i}`) % rivals.length) | 0] as Team
    const isHome = i % 2 === 0
    const home = isHome ? team.name : rival.name
    const away = isHome ? rival.name : team.name
    const p = mockPrediction(home, away)
    const score = p.top_scores[hash(`${home}${away}${i}`) % p.top_scores.length]?.score ?? '1-1'
    const [hg, ag] = score.split('-').map(Number)
    return { date: addDays(today, -(i * 7 + 3)), league: team.league, home, away, home_goals: hg, away_goals: ag }
  })
  const elo = ELO.get(team.name) ?? 1750
  return {
    team: team.name,
    league: team.league,
    results,
    ratings: { elo, pi_home: round((elo - 1750) / 200 + 0.3, 2), pi_away: round((elo - 1750) / 200 - 0.2, 2), form5: 1.6 },
  }
}
