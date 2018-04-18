import { AfterViewInit, Component, ElementRef, OnInit, ViewChild, Input, Output, EventEmitter } from '@angular/core';
import { ApiService } from '../_service/api.service';
import { UserEntityService } from '../_service/user-entity.service';
import { UserEntity } from '../_class/user-entity';
import { single, multi } from '../data';

import { GraphComponent } from '../graph/graph.component';

import * as shape from 'd3-shape';


@Component({
  selector: 'app-home',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  providers: [UserEntityService, ApiService],
  host: {
    '(window:resize)': 'onResize($event)'
  }
})
export class DashboardComponent implements OnInit, AfterViewInit {

  userEntities: UserEntity[] = [];

  newUserEntity: UserEntity = new UserEntity();

  connections: any[] = [
    {
      "name": "OnePlus ONEPLUS A5000",
      "series": [

      ]
    },
    {
      "name": "Nexus 5X",
      "series": [

      ]
    },
    {
      "name": "Samsung Galaxy S8",
      "series": [

      ]
    }
  ];

  data: any[] = [];

  theme = 'dark';
  chartType: string;
  chartGroups: any[];
  chart: any;
  realTimeData: boolean = false;
  countries: any[];
  single: any[];
  multi: any[];
  fiscalYearReport: any[];
  dateData: any[];
  dateDataWithRange: any[];
  calendarData: any[];
  statusData: any[];
  sparklineData: any[];
  timelineFilterBarData: any[];
  graph: { links: any[], nodes: any[] };
  bubble: any;
  linearScale: boolean = false;
  range: boolean = false;

  view: any[];
  width: number = 700;
  height: number = 300;
  fitContainer: boolean = false;

  // options
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  showLegend = true;
  legendTitle = 'Legend';
  showXAxisLabel = true;
  tooltipDisabled = false;
  xAxisLabel = 'Time';
  showYAxisLabel = true;
  yAxisLabel = 'Quantity';
  showGridLines = true;
  innerPadding = '10%';
  barPadding = 8;
  groupPadding = 16;
  roundDomains = false;
  maxRadius = 10;
  minRadius = 3;
  showSeriesOnHover = true;
  roundEdges: boolean = true;
  animations: boolean = true;
  xScaleMin: any;
  xScaleMax: any;
  yScaleMin: number;
  yScaleMax: number;

  curves = {
    Basis: shape.curveBasis,
    'Basis Closed': shape.curveBasisClosed,
    Bundle: shape.curveBundle.beta(1),
    Cardinal: shape.curveCardinal,
    'Cardinal Closed': shape.curveCardinalClosed,
    'Catmull Rom': shape.curveCatmullRom,
    'Catmull Rom Closed': shape.curveCatmullRomClosed,
    Linear: shape.curveLinear,
    'Linear Closed': shape.curveLinearClosed,
    'Monotone X': shape.curveMonotoneX,
    'Monotone Y': shape.curveMonotoneY,
    Natural: shape.curveNatural,
    Step: shape.curveStep,
    'Step After': shape.curveStepAfter,
    'Step Before': shape.curveStepBefore,
    default: shape.curveLinear
  };

  colorScheme = {
    //domain: ['#55d3f5', '#f43a59', '#F0F', '#32dbc3', '#2DAAE5']
    domain: ['#FF3333', '#FF33FF', '#9559fe',  '#CC33FF', '#0000FF', '#33CCFF', '#33FFFF', '#33FF66', '#CCFF33', '#FFCC00', '#FF6600']
  };

  // line, area
  autoScale = true;
  timeline = true;

  // gauge
  gaugeMin: number = 0;
  gaugeMax: number = 100;
  gaugeLargeSegments: number = 10;
  gaugeSmallSegments: number = 5;
  gaugeTextValue: string = '';
  gaugeUnits: string = 'alerts';
  gaugeAngleSpan: number = 240;
  gaugeStartAngle: number = -120;
  gaugeShowAxis: boolean = true;
  gaugeValue: number = 50; // linear gauge value
  gaugePreviousValue: number = 70;


  // line interpolation
  curveType: string = 'Linear';
  curve: any = this.curves[this.curveType];
  interpolationTypes = [
    'Basis', 'Bundle', 'Cardinal', 'Catmull Rom', 'Linear', 'Monotone X',
    'Monotone Y', 'Natural', 'Step', 'Step After', 'Step Before'
  ];

  closedCurveType: string = 'Linear Closed';
  closedCurve: any = this.curves[this.closedCurveType];
  closedInterpolationTypes = [
    'Basis Closed', 'Cardinal Closed', 'Catmull Rom Closed', 'Linear Closed'
  ];

  colorSets: any;
  schemeType: string = 'ordinal';
  selectedColorScheme: string;
  rangeFillOpacity: number = 0.15;

  @Output()
  add: EventEmitter<UserEntity> = new EventEmitter();

  @ViewChild('myIdentifier')
  myIdentifier: ElementRef;

  @ViewChild('graphWrapper') graphWrapper: ElementRef;

  constructor(
    private userEntityService: UserEntityService,
    private apiService: ApiService
  ) {
    Object.assign(this, { single, multi });
    this.fetch((data) => {
      this.rows = data;
      setTimeout(() => { this.loadingIndicator = false; }, 1500);
    });


  }

  public ngOnInit() {
    /*
    this.userEntityService
      .getAllUserEntities()
      .subscribe(
        (userEntities) => {
          this.userEntities = userEntities;
        }
      );
      */
    //this.view = [this.width, this.height];
    //var width = this.myIdentifier.nativeElement.offsetWidth;
    this.view = this.graphViewSize();
    console.log(this.graphWrapper);
  }

  ngAfterViewInit() {

  }

  graphViewSize() {
    return [window.innerWidth - 175, 500];
  }

  onResize(event) {
    this.view = this.graphViewSize();


    //this.graphWrapper.view = this.view;
    //this.graphWrapper.update();
    //console.log(this.graphWrapper);
  }

  consoleLog(obj) {
    console.log(obj);
    return obj;
  }

  getConnections(obj) {
    var connections = this.data.find(c => c.name === obj.series).series.find(c => c.name === obj.name && c.value == obj.value).connections;
    console.log("connections > ", connections);
    return connections;
  }

  processData(data: any[], index: number) {
    var dates = {};
    for (var key in data) {
      var connection = data[key];
      var connected_at = connection.connected_at;
      var k = `${connected_at}`;
      dates[k] = dates[k] || { count: 0, connections: [] };
      dates[k].count++;
      dates[k].connections.push(connection);
    }

    var count = 0;
    for (var date in dates) {
      var obj = dates[date];
      count = count + obj.count;
      //count = obj.count;
      this.connections[index]["series"].push({
        "value": count,
        "name": date,
        "label": '',
        "connections": obj.connections,
      });
    }
  }


  rows = [];
  loadingIndicator: boolean = true;
  reorderable: boolean = true;

  columns = [
    { prop: 'id' },
    { name: 'name' },
    { name: 'IMEI' },
    { name: 'token' }
  ];


  fetch(cb) {
    const req = new XMLHttpRequest();
    req.open('GET', `assets/data/profile.json`);

    req.onload = () => {
      cb(JSON.parse(req.response));
    };

    req.send();
  }

  fetchConnectionsAndreas(cb) {

    const request = new XMLHttpRequest();
    request.open('GET', `assets/data/connections_andreas.json`);

    request.onload = () => {
      cb(JSON.parse(request.response));
    };

    request.send();
  }

  fetchConnectionsAlex(cb) {

    const request = new XMLHttpRequest();
    request.open('GET', `assets/data/connections_alex.json`);

    request.onload = () => {
      cb(JSON.parse(request.response));
    };

    request.send();
  }

  selected = [];


  onSelect({ selected }) {
    console.log('Select Event', selected, this.selected);

    this.selected.splice(0, this.selected.length);
    this.selected.push(...selected);
  }

  onActivate(event) {
    console.log('Activate Event', event);
  }

  displayCheck(row) {
    //return row.name !== 'Ethel Price';
    return true;
  }

}
