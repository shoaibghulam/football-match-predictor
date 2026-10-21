import type { BucketStat } from '../../api/results'
import { formatCount, pct1 } from './labels'

/** "When we said 60–70%, it happened X%". */
export function CalibrationTable({ buckets }: { buckets: readonly BucketStat[] }) {
  const rows = buckets.filter((b) => b.n > 0)
  return (
    <section aria-labelledby="calibration" className="card flex flex-col gap-3 p-4">
      <h2 id="calibration" className="font-semibold">
        Calibration
      </h2>
      <p className="muted text-xs">When our favourite outcome had this probability, how often did it happen?</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="muted text-left text-xs">
            <th scope="col" className="py-1 font-medium">We said</th>
            <th scope="col" className="py-1 font-medium">Matches</th>
            <th scope="col" className="py-1 font-medium">Average said</th>
            <th scope="col" className="py-1 font-medium">It happened</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => (
            <tr key={b.bucket} className="border-t border-slate-100 dark:border-slate-800">
              <th scope="row" className="py-1.5 text-left font-medium tabular-nums">
                {b.bucket.replace('-', '–')}
              </th>
              <td className="tabular-nums">{formatCount(b.n)}</td>
              <td className="tabular-nums">{pct1(b.avg_prob)}</td>
              <td className="tabular-nums font-semibold">{pct1(b.accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted text-xs">Well calibrated means the last two columns are close.</p>
    </section>
  )
}
