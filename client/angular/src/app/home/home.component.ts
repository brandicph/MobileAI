import { Component, OnInit } from '@angular/core';
import { Chart, Highcharts } from 'angular-highcharts';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  chart = new Chart({
      title: {
        text: 'Linechart'
      },
      credits: {
        enabled: false
      },
      series: [{
        name: 'Line 1',
        type: 'line',
        data: [1, 2, 3]
      },{
        name: 'Line 2',
        type: 'arearange',
        data: [[1,5],[1,5],[1,5]]
      }]
    });

  constructor() { }

  ngOnInit() {
      console.log(this.chart.options.series[1]);
  }

  // add point to chart serie
  add() {
    this.chart.addPoint(Math.floor(Math.random() * 10));
  }

}
