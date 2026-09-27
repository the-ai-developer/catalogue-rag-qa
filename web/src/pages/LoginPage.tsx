import { createSignal } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { setApiKey } from '../api/client';

export default function LoginPage() {
  const [key, setKey] = createSignal('');
  const [error, setError] = createSignal<string | null>(null);
  const navigate = useNavigate();

  return (
    <form
      class="card form"
      onSubmit={(e) => {
        e.preventDefault();
        if (!key().trim()) {
          setError('API key is required');
          return;
        }
        setApiKey(key().trim());
        navigate('/catalogue');
      }}
    >
      <h2 class="page-title">Sign in</h2>
      <p class="muted">Use your catalogue API key (X-API-Key). Roles: viewer, editor, admin.</p>
      {error() && <div class="banner banner-error">{error()}</div>}
      <label>
        API key
        <input
          class="input" type="password" value={key()}
          onInput={(e) => setKey(e.currentTarget.value)}
          placeholder="X-API-Key" autocomplete="off"
        />
      </label>
      <button class="btn btn-primary">Continue</button>
    </form>
  );
}
