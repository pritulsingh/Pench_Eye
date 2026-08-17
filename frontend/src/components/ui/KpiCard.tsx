import React from 'react';
import { cn } from '@/lib/utils';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

interface KpiCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  loading?: boolean;
  className?: string;
}

export function KpiCard({ title, value, subtitle, icon, trend, loading, className }: KpiCardProps) {
  return (
    <div className={cn('kpi-card flex flex-col', className)}>
      <div className="flex items-start justify-between mb-2">
        <span className="text-sm font-medium text-muted-foreground">{title}</span>
        <div className="text-tiger-500 bg-tiger-500/10 p-2 rounded-md">
          {icon}
        </div>
      </div>
      
      {loading ? (
        <div className="animate-pulse space-y-2 mt-auto">
          <div className="h-8 bg-secondary/80 rounded w-1/2"></div>
          {subtitle && <div className="h-4 bg-secondary/80 rounded w-3/4"></div>}
        </div>
      ) : (
        <div className="mt-auto">
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-foreground tracking-tight">{value}</span>
            {trend && (
              <span className={cn(
                "flex items-center text-xs font-medium",
                trend.isPositive ? "text-green-400" : "text-red-400"
              )}>
                {trend.isPositive ? <ArrowUpRight className="w-3 h-3 mr-1" /> : <ArrowDownRight className="w-3 h-3 mr-1" />}
                {Math.abs(trend.value)}%
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
          )}
        </div>
      )}
    </div>
  );
}
