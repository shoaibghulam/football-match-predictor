import { useId, useState } from 'react'
import { MIN_TEAM_QUERY, useTeams } from '../api/hooks'
import type { Team } from '../api/types'
import { useDebouncedValue } from '../hooks/useDebouncedValue'

interface TeamComboboxProps {
  label: string
  value: string | null
  onSelect: (team: Team) => void
  placeholder?: string
}

const DEBOUNCE_MS = 250
const MAX_OPTIONS = 12

/** Searchable team picker following the ARIA 1.2 combobox pattern (list autocomplete). */
export function TeamCombobox({ label, value, onSelect, placeholder = 'Search teams…' }: TeamComboboxProps) {
  const inputId = useId()
  const listId = useId()
  const [query, setQuery] = useState('')
  const [editing, setEditing] = useState(false)
  const [active, setActive] = useState(-1)
  const debounced = useDebouncedValue(query, DEBOUNCE_MS)
  const { data, isFetching, isError } = useTeams(editing ? debounced : '')

  const options = (data ?? []).slice(0, MAX_OPTIONS)
  const term = query.trim()
  const open = editing && term.length >= MIN_TEAM_QUERY
  const optionId = (i: number) => `${listId}-opt-${i}`

  function choose(team: Team) {
    onSelect(team)
    setEditing(false)
    setQuery('')
    setActive(-1)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'ArrowDown' && open) {
      e.preventDefault()
      setActive((i) => Math.min(options.length - 1, i + 1))
    } else if (e.key === 'ArrowUp' && open) {
      e.preventDefault()
      setActive((i) => Math.max(0, i - 1))
    } else if (e.key === 'Enter' && open) {
      const team = options[active] ?? (options.length === 1 ? options[0] : undefined)
      if (team) {
        e.preventDefault()
        choose(team)
      }
    } else if (e.key === 'Escape') {
      setEditing(false)
      setQuery('')
    }
  }

  let status = ''
  if (open && isError) status = 'Could not search teams.'
  else if (open && options.length === 0 && !isFetching && debounced.trim() === term) status = 'No teams found.'
  else if (open && options.length === 0) status = 'Searching…'

  return (
    <div className="relative flex flex-col gap-1 text-sm">
      <label htmlFor={inputId} className="font-medium">
        {label}
      </label>
      <input
        id={inputId}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && active >= 0 ? optionId(active) : undefined}
        autoComplete="off"
        className="input"
        placeholder={placeholder}
        value={editing ? query : (value ?? '')}
        onFocus={() => setEditing(true)}
        onBlur={() => setEditing(false)}
        onChange={(e) => {
          setEditing(true)
          setQuery(e.target.value)
          setActive(-1)
        }}
        onKeyDown={handleKeyDown}
      />
      <ul
        id={listId}
        role="listbox"
        aria-label={`${label} suggestions`}
        hidden={!open || options.length === 0}
        className="card absolute top-full z-20 mt-1 max-h-72 w-full overflow-y-auto py-1"
      >
        {options.map((t, i) => (
          <li
            key={`${t.name}-${t.league}`}
            id={optionId(i)}
            role="option"
            aria-selected={i === active}
            className={`flex cursor-pointer justify-between gap-2 px-3 py-2 ${
              i === active ? 'bg-blue-100 dark:bg-blue-950' : 'hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
            onMouseDown={(e) => {
              e.preventDefault()
              choose(t)
            }}
          >
            <span>{t.name}</span>
            <span className="muted text-xs">{t.league}</span>
          </li>
        ))}
      </ul>
      {open && options.length === 0 && status && (
        <div className="card muted absolute top-full z-20 mt-1 w-full px-3 py-2">{status}</div>
      )}
      <span className="sr-only" aria-live="polite">
        {open && options.length > 0 ? `${options.length} teams found` : status}
      </span>
    </div>
  )
}
