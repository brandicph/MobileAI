import { Component, OnInit } from '@angular/core';
import { trigger, state, style, transition, animate, animateChild, keyframes, stagger, query, group } from '@angular/animations';
import * as $ from 'jquery';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  animations: [],
})
export class AppComponent implements OnInit {
  title = 'app';

  showTopNav = true;

  ngOnInit(){
    $(function() {

    });
  }

}
