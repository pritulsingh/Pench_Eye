import React from 'react';
import { Sidebar } from './Sidebar';
import TopBar from './TopBar';

interface LayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export default function Layout({ children, title, subtitle }: LayoutProps) {
  return (
    <div className="min-h-screen flex text-foreground">
      <Sidebar />
      <div className="flex-1 ml-16 lg:ml-60 flex flex-col relative min-h-screen">
        <TopBar />
        <main className="flex-1 p-6 lg:p-8 pb-12 max-w-[1600px] mx-auto w-full animate-fade-in">
          <header className="mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
                {subtitle && (
                  <p className="text-muted-foreground mt-1 text-sm font-medium">{subtitle}</p>
                )}
              </div>
              <div className="hidden sm:block" />
            </div>
          </header>
          {children}
        </main>
      </div>
    </div>
  );
}
