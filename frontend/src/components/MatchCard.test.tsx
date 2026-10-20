import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { makeFixture, makePrediction } from '../test/factories'
import { MatchCard } from './MatchCard'

describe('MatchCard', () => {
  it('shows teams, kickoff, probabilities and prediction stats', () => {
    render(<MatchCard fixture={makeFixture()} />)
    const card = screen.getByRole('article', { name: 'Arsenal vs Chelsea' })
    expect(within(card).getByText('15:00')).toBeInTheDocument()
    expect(within(card).getByRole('img', { name: /Arsenal 52%, draw 26%, Chelsea 22%/ })).toBeInTheDocument()
    expect(within(card).getByText('1.6 – 1.0')).toBeInTheDocument()
    expect(within(card).getByText('49%')).toBeInTheDocument()
    expect(within(card).getByText('47%')).toBeInTheDocument()
    expect(within(card).getByText('uses market odds')).toBeInTheDocument()
  })

  it('shows only the top 3 scorelines', () => {
    render(<MatchCard fixture={makeFixture()} />)
    const list = screen.getByRole('list', { name: 'Most likely scores' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(3)
    expect(within(list).getByText('1-0')).toBeInTheDocument()
    expect(within(list).queryByText('0-0')).not.toBeInTheDocument()
  })

  it('describes the favourite without implying certainty', () => {
    render(<MatchCard fixture={makeFixture()} />)
    expect(screen.getByText(/Favourite:/)).toHaveTextContent('Favourite: Arsenal (52%) — not a certainty.')
  })

  it('names a draw or away favourite correctly', () => {
    const { rerender } = render(
      <MatchCard fixture={makeFixture({ prediction: makePrediction({ probs: { home: 0.3, draw: 0.38, away: 0.32 } }) })} />,
    )
    expect(screen.getByText(/not a certainty/)).toHaveTextContent('A draw (38%)')
    rerender(<MatchCard fixture={makeFixture({ prediction: makePrediction({ probs: { home: 0.2, draw: 0.2, away: 0.6 } }) })} />)
    expect(screen.getByText(/not a certainty/)).toHaveTextContent('Chelsea (60%)')
  })

  it('shows the final score for finished matches', () => {
    render(<MatchCard fixture={makeFixture({ status: 'finished', score: { home: 2, away: 1 } })} />)
    expect(screen.getByText('Full time')).toBeInTheDocument()
    expect(screen.getByText('Final score Arsenal 2, Chelsea 1')).toBeInTheDocument()
    expect(screen.getByText('Model view (current ratings)')).toBeInTheDocument()
  })

  it('labels a frozen pre-kickoff snapshot as the pre-match prediction', () => {
    const fixture = makeFixture({
      status: 'finished',
      score: { home: 2, away: 1 },
      prediction_frozen_at: '2026-10-21T18:30:00Z',
    })
    render(<MatchCard fixture={fixture} />)
    const label = screen.getByText('Pre-match prediction')
    expect(label).toHaveAttribute('title', expect.stringContaining('Frozen before kickoff'))
  })

  it('shows the date in multi-day views and handles unknown times', () => {
    render(<MatchCard fixture={makeFixture({ kickoff: '2026-10-25' })} showDate />)
    expect(screen.getByText(/Time TBC/)).toBeInTheDocument()
  })

  it('handles a missing prediction and the ratings-only model', () => {
    const { rerender } = render(<MatchCard fixture={makeFixture({ prediction: null })} />)
    expect(screen.getByText(/No prediction available/)).toBeInTheDocument()
    rerender(<MatchCard fixture={makeFixture({ prediction: makePrediction({ model: 'base' }) })} />)
    expect(screen.getByText('ratings only')).toBeInTheDocument()
  })

  it('calls onOpen when Details is activated', async () => {
    const onOpen = vi.fn()
    const fixture = makeFixture()
    render(<MatchCard fixture={fixture} onOpen={onOpen} />)
    await userEvent.click(screen.getByRole('button', { name: 'Details: Arsenal vs Chelsea' }))
    expect(onOpen).toHaveBeenCalledWith(fixture)
  })
})
