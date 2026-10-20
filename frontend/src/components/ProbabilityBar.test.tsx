import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ProbabilityBar } from './ProbabilityBar'

describe('ProbabilityBar', () => {
  it('renders three segments with widths that sum to 100%', () => {
    render(<ProbabilityBar probs={{ home: 0.456, draw: 0.267, away: 0.277 }} homeLabel="Arsenal" awayLabel="Chelsea" />)
    const widths = ['home', 'draw', 'away'].map((o) => parseFloat(screen.getByTestId(`bar-${o}`).style.width))
    expect(widths.reduce((a, b) => a + b, 0)).toBe(100)
  })

  it('exposes an accessible summary', () => {
    render(<ProbabilityBar probs={{ home: 0.52, draw: 0.26, away: 0.22 }} homeLabel="Arsenal" awayLabel="Chelsea" />)
    expect(screen.getByRole('img')).toHaveAccessibleName(
      'Win probabilities: Arsenal 52%, draw 26%, Chelsea 22%',
    )
  })

  it('emphasises the favourite in the legend and dims the others', () => {
    render(<ProbabilityBar probs={{ home: 0.2, draw: 0.3, away: 0.5 }} homeLabel="A" awayLabel="B" />)
    expect(screen.getByText('50%')).toHaveClass('font-bold')
    expect(screen.getByText('20%')).not.toHaveClass('font-bold')
    expect(screen.getByTestId('bar-home')).toHaveClass('opacity-70')
    expect(screen.getByTestId('bar-away')).not.toHaveClass('opacity-70')
  })

  it('omits zero segments and the legend in compact mode', () => {
    render(<ProbabilityBar probs={{ home: 1, draw: 0, away: 0 }} homeLabel="A" awayLabel="B" compact />)
    expect(screen.queryByTestId('bar-draw')).not.toBeInTheDocument()
    expect(screen.queryByText('100%')).not.toBeInTheDocument()
  })
})
