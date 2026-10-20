import type { RouteObject } from 'react-router'
import { Layout } from './components/layout/Layout'
import { AboutPage } from './pages/AboutPage'
import { MatchesPage } from './pages/MatchesPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { PredictPage } from './pages/PredictPage'
import { TeamPage } from './pages/TeamPage'

export const routes: RouteObject[] = [
  {
    element: <Layout />,
    children: [
      { index: true, element: <MatchesPage /> },
      { path: 'team', element: <TeamPage /> },
      { path: 'team/:name', element: <TeamPage /> },
      { path: 'predict', element: <PredictPage /> },
      { path: 'about', element: <AboutPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]
