import type { RouteDefinition } from '@solidjs/router';
import App from './App';
import LoginPage from './pages/LoginPage';
import CataloguePage from './pages/CataloguePage';
import ItemPage from './pages/ItemPage';
import AskPage from './pages/AskPage';
import DescriptionsPage from './pages/DescriptionsPage';

export const routes: RouteDefinition[] = [
  {
    path: '/',
    component: App,
    children: [
      { path: '/', component: CataloguePage },
      { path: '/catalogue', component: CataloguePage },
      { path: '/catalogue/:id', component: ItemPage },
      { path: '/ask', component: AskPage },
      { path: '/descriptions', component: DescriptionsPage },
      { path: '/login', component: LoginPage },
    ],
  },
];
