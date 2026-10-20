import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { App } from './App'
import { createQueryClient } from './api/queryClient'
import { renderRoute } from './test/render'

describe('App shell', () => {
  it('renders navigation, footer attribution and navigates to About', async () => {
    const { user } = renderRoute('/')
    const nav = screen.getByRole('navigation', { name: 'Main' })
    expect(nav).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Matches' })).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('contentinfo')).toHaveTextContent('football-data.co.uk')

    await user.click(screen.getByRole('link', { name: 'About' }))
    expect(screen.getByRole('heading', { level: 1, name: 'How accurate is this?' })).toBeInTheDocument()
    expect(screen.getByText(/not betting advice/i, { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText(/238,000 matches/)).toBeInTheDocument()
    expect(screen.getByText('≈50.5%')).toBeInTheDocument()
  })

  it('mounts the real App with browser routing and query client', async () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1, name: 'Matches' })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText('Loading matches…')).not.toBeInTheDocument())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(createQueryClient().getDefaultOptions().queries?.staleTime).toBe(5 * 60 * 1000)
  })

  it('shows a not-found page for unknown routes', () => {
    renderRoute('/nope')
    expect(screen.getByRole('heading', { name: 'Page not found' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to matches' })).toHaveAttribute('href', '/')
  })
})
