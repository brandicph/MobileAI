import { AuthGuard } from './_common/auth.guard';
import { AppLayoutComponent } from './_layout/app-layout/app-layout.component';
import { LoginComponent } from './login/login.component';
import { DashboardComponent } from './dashboard/dashboard.component';

export const APP_ROUTES = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  //{path: 'find', redirectTo: 'search'},
  {
      path: '',
      component: AppLayoutComponent,
      children: [
        { path: 'dashboard', component: DashboardComponent },
      ]
  },
  { path: '**', component: LoginComponent }
];
