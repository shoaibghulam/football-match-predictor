import { NavLink, Outlet } from 'react-router'

const NAV = [
  { to: '/', label: 'Matches', end: true },
  { to: '/team', label: 'Teams', end: false },
  { to: '/predict', label: 'Predict', end: false },
  { to: '/results', label: 'Results', end: false },
  { to: '/about', label: 'About', end: false },
]

function navClass({ isActive }: { isActive: boolean }): string {
  const base = 'rounded-lg px-3 py-2 text-sm font-medium'
  return isActive
    ? `${base} bg-blue-700 text-white dark:bg-blue-400 dark:text-slate-950`
    : `${base} text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800`
}

export function Layout() {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only z-50 rounded bg-white px-3 py-2 focus:not-sr-only focus:absolute focus:left-2 focus:top-2 dark:bg-slate-900"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 py-2">
          <NavLink to="/" className="flex items-center gap-2 text-lg font-bold tracking-tight">
            <span aria-hidden="true">⚽</span> MatchPredict
          </NavLink>
          <nav aria-label="Main">
            <ul className="flex gap-1">
              {NAV.map((n) => (
                <li key={n.to}>
                  <NavLink to={n.to} end={n.end} className={navClass}>
                    {n.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        <Outlet />
      </main>
      <footer className="border-t border-slate-200 py-6 text-xs dark:border-slate-800">
        <div className="muted mx-auto flex max-w-5xl flex-col gap-1 px-4">
          <p>
            Match data and odds from{' '}
            <a
              href="https://www.football-data.co.uk/"
              className="underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              football-data.co.uk
            </a>
            . Probabilities are model estimates, for entertainment and information only — not betting advice.
          </p>
          <p>
            Gambling can be addictive. If it stops being fun, visit{' '}
            <a href="https://www.begambleaware.org/" className="underline" target="_blank" rel="noopener noreferrer">
              BeGambleAware.org
            </a>
            . 18+.
          </p>
        </div>
      </footer>
    </div>
  )
}
