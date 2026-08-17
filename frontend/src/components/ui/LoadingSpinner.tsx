interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  label?: string
  className?: string
}

const sizeClasses = { sm: 'h-4 w-4', md: 'h-6 w-6', lg: 'h-10 w-10' }

export function LoadingSpinner({ size = 'md', label, className }: LoadingSpinnerProps) {
  return (
    <div className={`flex flex-col items-center justify-center gap-3 ${className ?? ''}`}>
      <svg
        className={`animate-spin text-tiger-700 ${sizeClasses[size]}`}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      {label && <p className="text-sm text-muted-foreground">{label}</p>}
    </div>
  )
}