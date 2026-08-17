import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'tiger' | 'demo' | 'success' | 'warning' | 'error' | 'review' | 'new'

interface BadgeProps {
  children: React.ReactNode
  variant?: BadgeVariant
  className?: string
}

const variantClasses: Record<BadgeVariant, string> = {
  default: 'bg-secondary text-secondary-foreground border border-border',
  tiger: 'bg-tiger-100 text-tiger-800 border border-tiger-200',
  demo: 'bg-amber-50 text-amber-800 border border-amber-200',
  success: 'bg-green-50 text-green-800 border border-green-200',
  warning: 'bg-yellow-50 text-yellow-800 border border-yellow-200',
  error: 'bg-red-50 text-red-800 border border-red-200',
  review: 'bg-blue-50 text-blue-800 border border-blue-200',
  new: 'bg-orange-50 text-orange-800 border border-orange-200',
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full', variantClasses[variant], className)}>
      {children}
    </span>
  )
}