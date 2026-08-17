import { Link } from 'react-router-dom';
import { Compass } from 'lucide-react';

import { EmptyState } from '@/components/ui/EmptyState';

export default function NotFound() {
  return (
    <EmptyState
      icon={<Compass />}
      title="This page does not exist"
      description="The route you followed is not part of Pench Eye. Head back to the command center."
      action={
        <Link to="/" className="btn-primary">
          Go to dashboard
        </Link>
      }
    />
  );
}
