import React from 'react';
import { cn } from '@/lib/utils';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const getStatusConfig = (s: string) => {
    switch (s.toLowerCase()) {
      case 'active':
      case 'approved':
      case 'auto_match':
      case 'human_verified':
        return { color: 'bg-green-600', text: 'text-green-800', bg: 'bg-green-50', border: 'border-green-200' };
      case 'pending':
      case 'maintenance':
      case 'review_required':
      case 'new_individual':
        return { color: 'bg-amber-500', text: 'text-amber-800', bg: 'bg-amber-50', border: 'border-amber-200' };
      case 'inactive':
      case 'rejected':
      case 'quarantined':
      case 'deleted':
        return { color: 'bg-red-500', text: 'text-red-800', bg: 'bg-red-50', border: 'border-red-200' };
      default:
        return { color: 'bg-gray-500', text: 'text-gray-700', bg: 'bg-gray-50', border: 'border-gray-200' };
    }
  };

  const config = getStatusConfig(status);

  return (
    <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border', config.bg, config.text, config.border, className)}>
      <span className={cn('w-1.5 h-1.5 rounded-full', config.color)} />
      {status.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
    </span>
  );
}
