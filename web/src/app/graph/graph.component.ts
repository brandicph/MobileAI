import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { single, multi } from '../data';



@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss'],
  providers: []
})
export class GraphComponent implements OnInit {

  single: any[];
  multi: any[];

  view: any[] = [700, 400];

  // options
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  showLegend = true;
  showXAxisLabel = true;
  xAxisLabel = 'Country';
  showYAxisLabel = true;
  yAxisLabel = 'Population';

  colorScheme = {
    domain: ['#55d3f5', '#f43a59', '#F0F', '#32dbc3', '#2DAAE5']
  };

  // line, area
  autoScale = true;


  onSelect(event) {
    console.log(event);
  }

  constructor(

  ) {
    Object.assign(this, { single, multi })
  }

  ngOnInit() {

    //this.applyDimensions();
  }

}
