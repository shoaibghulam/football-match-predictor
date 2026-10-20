import { Link } from 'react-router'

export function NotFoundPage() {
  return (
    <div className="flex flex-col items-start gap-3">
      <h1 className="text-2xl font-bold">Page not found</h1>
      <p className="muted">That page doesn&apos;t exist.</p>
      <Link to="/" className="btn btn-primary">
        Back to matches
      </Link>
    </div>
  )
}
