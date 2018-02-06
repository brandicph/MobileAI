import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpModule } from '@angular/http';
import { RouterModule } from '@angular/router';
import { NgModule } from '@angular/core';
import { ChartModule, HIGHCHARTS_MODULES } from 'angular-highcharts';
import highstock from 'highcharts/modules/stock.src'; 
import more from 'highcharts/highcharts-more.src';
import exporting from 'highcharts/modules/exporting.src';

export function highchartsModules() {
  // apply Highcharts Modules to this array
  return [ more, exporting, highstock ];
}


import { AppComponent } from './app.component'
import { HomeComponent } from './home/home.component'

import { APP_ROUTES } from './routes';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
  ],
  imports: [
    ChartModule,
    BrowserModule,
    BrowserAnimationsModule,
    HttpModule,
    RouterModule.forRoot(APP_ROUTES)
  ],
  providers: [ { provide: HIGHCHARTS_MODULES, useFactory: highchartsModules }],
  bootstrap: [AppComponent]
})
export class AppModule { }
