// preview.jsx
import React from 'react';
import ColorPreview from '@/components/pub/ColorPreview';
import { NODE_ENV } from '@/config/env';

export default function Preview() {
  return (
    <div className="bg-gray-50 h-[90%]">
      <section className="mb-8">
        <div className="p-6">
          <h1>Preview</h1>
          <ColorPreview />
        </div>
      </section>
    </div>
  );
}
