import React from 'react'

interface KpiProps {
  label: string
  value: string | number
  trend?: string
  small?: boolean
}

export default function Kpi({ label, value, trend, small }: KpiProps) {
  return (
    <div className={`kpi-card ${small ? 'p-3 text-sm' : ''}`}>
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <div className="text-muted-foreground text-xs font-semibold">{label}</div>
          <div className="text-2xl font-bold tracking-tight">{value}</div>
        </div>
        {trend && <div className="text-sm text-muted-foreground">{trend}</div>}
      </div>
    </div>
  )
}
