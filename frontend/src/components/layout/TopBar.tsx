import React from 'react'
import { Bell, Wifi, Clock } from 'lucide-react'
import { useApi } from '@/hooks/useApi'
import { dashboardApi } from '@/api/client'

export default function TopBar() {
  const { data: stats } = useApi(() => dashboardApi.getStats())

  return (
    <div className="w-full border-b border-border bg-white/60 backdrop-blur-sm">
      <div className="max-w-[1600px] mx-auto px-4 lg:px-8 h-16 flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="hidden lg:block">
            <div className="text-sm font-semibold">Command Center</div>
            <div className="text-xs text-muted-foreground">Pench Tiger Reserve · Live Operations</div>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-green-600 inline-block" />
              <span>All systems operational</span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-3 text-xs text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>{new Date().toLocaleString()}</span>
          </div>
          <div className="relative">
            <button className="p-2 rounded-md hover:bg-secondary/50">
              <Bell className="w-5 h-5" />
            </button>
            <span className="absolute -top-1 -right-1 bg-amber-500 text-white text-[10px] font-bold px-1.5 rounded-full">3</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-sm font-medium">Control Room</div>
            <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center text-sm">CR</div>
          </div>
        </div>
      </div>
    </div>
  )
}
