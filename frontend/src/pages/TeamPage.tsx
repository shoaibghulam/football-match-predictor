import { useNavigate, useParams } from 'react-router'
import { TeamCombobox } from '../components/TeamCombobox'
import { TeamFormPanel } from '../components/team/TeamFormPanel'
import { UpcomingFixtures } from '../components/team/UpcomingFixtures'

export function TeamPage() {
  const { name } = useParams()
  const navigate = useNavigate()
  const team = name?.trim() || null

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{team ?? 'Teams'}</h1>
        <p className="muted text-sm">
          {team
            ? 'Upcoming fixtures with win chances, recent results and current ratings.'
            : "Pick a team to see its next matches and how likely it is to win them."}
        </p>
      </div>

      <div className="max-w-md">
        <TeamCombobox
          label={team ? 'Choose another team' : 'Choose a team'}
          value={null}
          placeholder="Type at least 2 letters, e.g. Liverpool"
          onSelect={(t) => void navigate(`/team/${encodeURIComponent(t.name)}`)}
        />
      </div>

      {team && (
        <>
          <section aria-labelledby="upcoming-heading" className="flex flex-col gap-3">
            <h2 id="upcoming-heading" className="text-lg font-semibold">
              Upcoming fixtures
            </h2>
            <UpcomingFixtures team={team} />
          </section>
          <TeamFormPanel team={team} />
        </>
      )}
    </div>
  )
}
