import React from 'react';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center p-8 text-center rounded-lg border border-dashed border-border bg-secondary/20", className)}>
      <div className="text-muted-foreground mb-4 opacity-80">
        {React.cloneElement(icon as React.ReactElement, { size: 48 })}
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground mb-6 max-w-sm">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
