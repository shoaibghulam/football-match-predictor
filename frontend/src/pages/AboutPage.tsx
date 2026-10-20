const BENCHMARKS = [
  { label: 'Always pick the home team', value: '≈45%' },
  { label: 'Draws in our data', value: '≈26%' },
  { label: 'Bookmaker favourite (our data)', value: '≈50.5%' },
  { label: 'Best published models', value: '49–54%' },
]

export function AboutPage() {
  return (
    <article className="mx-auto flex max-w-3xl flex-col gap-6 text-[15px] leading-relaxed">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">How accurate is this?</h1>
        <p className="muted">Honest answers about what these numbers mean, and what they don&apos;t.</p>
      </header>

      <section aria-labelledby="calibrated">
        <h2 id="calibrated" className="mb-2 text-lg font-semibold">
          Probabilities, not certainties
        </h2>
        <p>
          Every match shows three <strong>calibrated probabilities</strong>: home win, draw and away win. Calibrated
          means that, across many matches, outcomes we give 60% happen about 60% of the time. It also means that a
          60% favourite <em>fails to win</em> 4 times in 10. A highlighted favourite is just the most likely
          outcome, never a sure thing.
        </p>
      </section>

      <section aria-labelledby="accuracy">
        <h2 id="accuracy" className="mb-2 text-lg font-semibold">
          Realistic accuracy
        </h2>
        <p className="mb-3">
          Football is low-scoring and noisy, so nobody picks winners reliably. If you simply back the most likely
          outcome, expect to be right about <strong>half the time</strong>: roughly 50% for home/draw/away.
          Bookmakers&apos; pre-match odds, the hardest benchmark there is, pick the right outcome about 50.5% of the
          time on our data. Draws happen in about 26% of matches but are almost never the single most likely
          result, which is why we show all three probabilities instead of a single pick.
        </p>
        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {BENCHMARKS.map((b) => (
            <div key={b.label} className="card p-3">
              <dt className="muted text-xs">{b.label}</dt>
              <dd className="text-xl font-bold tabular-nums">{b.value}</dd>
            </div>
          ))}
        </dl>
        <p className="muted mt-2 text-sm">
          3-way accuracy on held-out seasons. We also track the ranked probability score (RPS), where good models
          land around 0.20–0.21.
        </p>
      </section>

      <section aria-labelledby="model">
        <h2 id="model" className="mb-2 text-lg font-semibold">
          How the model works
        </h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            <strong>Team ratings:</strong> our own Elo ratings and pi-ratings (separate home and away strength),
            updated after every result, plus recent form and rest days.
          </li>
          <li>
            <strong>Outcome model:</strong> a CatBoost gradient-boosting classifier on those ratings, with
            temperature scaling so the probabilities stay calibrated.
          </li>
          <li>
            <strong>Goals model:</strong> a Dixon-Coles style Poisson model that gives expected goals, likely
            scorelines, over 2.5 goals and both-teams-to-score.
          </li>
          <li>
            <strong>Market odds:</strong> when pre-match odds exist, a second model also uses them (badge
            “uses market odds”); otherwise you see “ratings only”.
          </li>
          <li>
            Trained on about <strong>238,000 matches from 2000 to 2026 across 38 leagues</strong>, and evaluated
            walk-forward, so it is always tested on seasons it has not seen.
          </li>
        </ul>
      </section>

      <section aria-labelledby="data">
        <h2 id="data" className="mb-2 text-lg font-semibold">
          Data
        </h2>
        <p>
          Results, fixtures and odds come from{' '}
          <a
            className="font-medium text-blue-700 underline dark:text-blue-400"
            href="https://www.football-data.co.uk/"
            target="_blank"
            rel="noopener noreferrer"
          >
            football-data.co.uk
          </a>
          , updated daily. Fixtures are usually published a few days before each round, so later dates may be empty.
        </p>
      </section>

      <section
        aria-labelledby="responsible"
        className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
      >
        <h2 id="responsible" className="mb-2 text-lg font-semibold">
          Responsible use
        </h2>
        <p>
          MatchPredict is for <strong>entertainment and information only. It is not betting advice.</strong> No
          model can make betting reliably profitable, and past accuracy does not guarantee future results. If you
          bet, only stake what you can afford to lose. For free, confidential support visit{' '}
          <a className="underline" href="https://www.begambleaware.org/" target="_blank" rel="noopener noreferrer">
            BeGambleAware.org
          </a>
          . 18+ only.
        </p>
      </section>
    </article>
  )
}
