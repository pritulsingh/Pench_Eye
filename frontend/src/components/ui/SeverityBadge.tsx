import { cn } from '@/lib/utils';

const SEVERITY_STYLES: Record<string, string> = {
  critical: 'bg-red-50 text-red-800 border-red-200',
  high: 'bg-orange-50 text-orange-800 border-orange-200',
  medium: 'bg-yellow-50 text-yellow-800 border-yellow-200',
  low: 'bg-blue-50 text-blue-800 border-blue-200',
  info: 'bg-secondary text-secondary-foreground border-border',
};

export function SeverityBadge({ severity, className }: { severity: string; className?: string }) {
  const style = SEVERITY_STYLES[severity.toLowerCase()] ?? SEVERITY_STYLES.info;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border shrink-0',
        style,
        className
      )}
    >
      {severity}
    </span>
  );
}
