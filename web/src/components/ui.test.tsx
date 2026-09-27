import { render, fireEvent } from '@solidjs/testing-library';
import { describe, expect, it, vi } from 'vitest';
import { CitationList, SpecForm } from './ui';
import type { Citation } from '../api/types';

const citations: Citation[] = [
  {
    sentence_index: 0, item_id: 'a', sku: 'KX-1042', chunk_id: 'c1', asset_id: null,
    modality: 'text', score: 0.83, snippet: 'The Kestrel bottle holds 12 oz.',
  },
];

describe('CitationList (Project 1 gate)', () => {
  it('refuses to render ungrounded answers', () => {
    const { container, unmount } = render(() => (
      <CitationList citations={citations} passed={false} />
    ));
    expect(container.textContent).toContain('Ungrounded answer withheld');
    expect(container.textContent).not.toContain('The Kestrel bottle holds 12 oz');
    unmount();
  });

  it('renders evidence when the citation check passed', () => {
    const { container, unmount } = render(() => (
      <CitationList citations={citations} passed={true} />
    ));
    expect(container.textContent).toContain('The Kestrel bottle holds 12 oz');
    expect(container.textContent).toContain('KX-1042');
    unmount();
  });
});

describe('SpecForm (Project 2 input)', () => {
  it('requires category before submitting', async () => {
    const onSubmit = vi.fn();
    const { container, unmount } = render(() => <SpecForm pending={false} onSubmit={onSubmit} />);
    await fireEvent.submit(container.querySelector('form')!);
    await new Promise((r) => setTimeout(r, 0)); // let Solid flush the signal
    expect(onSubmit).not.toHaveBeenCalled();
    expect(container.textContent).toContain('category is required');
    unmount();
  });

  it('submits a structured spec', async () => {
    const onSubmit = vi.fn();
    const { container, unmount } = render(() => <SpecForm pending={false} onSubmit={onSubmit} />);
    const inputs = container.querySelectorAll('input');
    await fireEvent.input(inputs[0], { target: { value: 'Kestrel Bottle' } });
    await fireEvent.input(inputs[1], { target: { value: 'drinkware' } });
    await fireEvent.input(inputs[3], { target: { value: 'height_cm=26' } });
    await fireEvent.input(inputs[4], { target: { value: 'leak-proof lid, dishwasher safe' } });
    await fireEvent.click(container.querySelector('button')!);
    expect(onSubmit).toHaveBeenCalledOnce();
    const [spec, beam, maxLen] = onSubmit.mock.calls[0];
    expect(spec.category).toBe('drinkware');
    expect(spec.dimensions).toEqual({ height_cm: 26 });
    expect(spec.features).toEqual(['leak-proof lid', 'dishwasher safe']);
    expect(beam).toBe(4);
    expect(maxLen).toBe(192);
    unmount();
  });
});
