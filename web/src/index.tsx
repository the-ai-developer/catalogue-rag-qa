/* @refresh reload */
import { render } from 'solid-js/web';
import { Router } from '@solidjs/router';
import { routes } from './routes';
import './styles/tokens.css';
import './styles/app.css';

const root = document.getElementById('root');
if (!root) throw new Error('missing #root element');

render(() => <Router>{routes}</Router>, root);
