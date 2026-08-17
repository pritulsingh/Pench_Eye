import React from 'react'

export default function SearchBar({ placeholder = 'Search' }: { placeholder?: string }) {
  return (
    <div className="relative w-full lg:w-72">
      <input
        className="filter-input w-full pl-3 pr-10"
        placeholder={placeholder}
        aria-label={placeholder}
      />
      <div className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">⌕</div>
    </div>
  )
}
