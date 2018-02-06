import { HomeComponent } from './home/home.component';

export const APP_ROUTES = [
  {path: '', redirectTo: 'home', pathMatch: 'full'},
  {path: 'find', redirectTo: 'search'},
  {path: 'home', component: HomeComponent},
  /*
  {
    path: 'artist/:artistId',
    component: null,
    children: [
      {path: '', redirectTo: 'tracks'}, 
      {path: 'tracks', component: null}, 
      {path: 'albums', component: null}, 
    ]
  },
  */
  {path: '**', component: HomeComponent}
];