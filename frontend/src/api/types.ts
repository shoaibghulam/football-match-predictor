/** Types for the MatchPredict API v1 (see docs/04-api.md). Optional fields are tolerated. */

export type Outcome = 'home' | 'draw' | 'away'

export interface Probs {
  home: number
  draw: number
  away: number
}

export interface ExpectedGoals {
  home: number
  away: number
}

export interface ScoreProb {
  /** "2-1" (home-away) */
  score: string
  p: number
}

/**
 * "market" when bookmaker odds were used as an input, "base" for ratings only,
 * "europe" for the cross-league UEFA model.
 */
export type ModelKind = 'market' | 'base' | 'europe'

export interface Prediction {
  probs: Probs
  expected_goals: ExpectedGoals
  over_2_5: number
  btts: number
  top_scores: ScoreProb[]
  model: ModelKind | (string & {})
  model_version: string
  /** "low" when a club is rated from a prior / has few rated games (European model and model v2). */
  confidence?: 'high' | 'low' | null
}

export interface Odds {
  home?: number | null
  draw?: number | null
  away?: number | null
}

export type FixtureStatus = 'scheduled' | 'live' | 'finished' | (string & {})

export type CompetitionType = 'domestic' | 'europe'

export interface LiveInfo {
  /** Match minute (from the feed, or estimated from kick-off). */
  minute: number | null
  minute_estimated?: boolean
  score: Score
}

export interface Score {
  home: number
  away: number
}

export interface Fixture {
  id: number | string
  /** League code, e.g. "E0" */
  league: string
  /** ISO datetime (or ISO date when the kickoff time is unknown) */
  kickoff: string | null
  /** Published (UK) match date, yyyy-mm-dd; used when the kickoff time is unknown. */
  date?: string
  home: string
  away: string
  odds?: Odds | null
  prediction: Prediction | null
  status?: FixtureStatus
  /** Final score, or the current score while live. */
  score?: Score | null
  competition_type?: CompetitionType
  /** UEFA stage: qual | league | ko | final */
  stage?: string | null
  /** UEFA match, or both clubs rated highly. */
  featured?: boolean
  live?: LiveInfo | null
  /** When set, `prediction` is the snapshot frozen before kickoff (UTC time of its last refresh). */
  prediction_frozen_at?: string | null
}

export interface League {
  code: string
  name: string
  country: string
}

export interface Team {
  name: string
  league: string
}

export interface Health {
  status: string
  model_version?: string
  last_ingest?: string | null
  matches?: number
  fixtures?: number
  europe_model_loaded?: boolean
  europe_model_version?: string | null
  football_data_org_key_configured?: boolean
  live_matches?: number
  model_v2_loaded?: boolean
  model_v2_version?: string | null
  /** Scopes served by model v2: "domestic" and/or "europe". */
  model_v2_scopes?: string[]
  /** First-start progress (absent on older backends = ready). */
  setup?: SetupStatus | null
}

export type SetupStage = 'starting' | 'seeding_history' | 'seeding_uefa' | 'ingesting' | 'ready' | (string & {})

export interface SetupStatus {
  stage: SetupStage
  detail?: string
  /** 0..1 within the stage, when known */
  progress?: number | null
  started_at?: string | null
  updated_at?: string | null
  first_start?: boolean
}

/** One past result. The backend shape may vary, so every field is optional. */
export interface FormResult {
  date?: string
  league?: string
  home?: string
  away?: string
  home_goals?: number | null
  away_goals?: number | null
  opponent?: string
  venue?: string
  goals_for?: number | null
  goals_against?: number | null
  result?: string
}

export interface TeamForm {
  team?: string
  name?: string
  league?: string
  results?: FormResult[]
  matches?: FormResult[]
  last_results?: FormResult[]
  ratings?: Record<string, number | null | undefined>
}

export interface ApiErrorBody {
  error: { code: string; message: string }
}
