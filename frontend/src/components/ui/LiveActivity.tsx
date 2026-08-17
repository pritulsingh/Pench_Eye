import React from 'react'
import { useApi } from '@/hooks/useApi'
import { dashboardApi } from '@/api/client'
import { relativeTime } from '@/lib/utils'

export default function LiveActivity() {
  const { data: stats } = useApi(() => dashboardApi.getStats())
  const events = (stats as any)?.live_activity ?? []

  return (
    <div className="card p-4 space-y-3 max-h-[520px] overflow-y-auto">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold">Live Activity</h3>
          <p className="text-xs text-muted-foreground">Recent detections and system events</p>
        </div>
        <a href="#/activity" className="text-xs text-tiger-700 hover:underline">
          View all
        </a>
      </div>

      <div className="divide-y divide-border/60">
        {(events.length ?? 0) === 0 ? (
          <div className="p-4 text-sm text-muted-foreground">No recent activity.</div>
        ) : (
          events.map((e: any) => (
            <div key={e.id} className="py-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 flex items-center justify-center rounded bg-secondary/50 text-sm">
                  {e.icon ?? '•'}
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium">{e.title}</div>
                  <div className="text-xs text-muted-foreground">{e.subtitle}</div>
                </div>
                <div className="text-xs text-muted-foreground">{relativeTime(e.ts)}</div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
