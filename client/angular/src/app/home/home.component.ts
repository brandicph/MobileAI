import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { Chart } from 'angular-highcharts';
import { UserEntityService } from '../_service/user-entity.service';
import { UserEntity } from '../_class/user-entity';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  providers: [UserEntityService]
})
export class HomeComponent implements OnInit {

  chart = new Chart({
      title: {
        text: 'RSRP'
      },
      credits: {
        enabled: false
      },
      series: [{
        name: 'Line 1',
        type: 'line',
        data: [1, 2, 3]
      }]
    });

  userEntities: UserEntity[] = [];

  newUserEntity: UserEntity = new UserEntity();

  @Output()
  add: EventEmitter<UserEntity> = new EventEmitter();

  constructor(
    private userEntityService: UserEntityService
  ) {
  }

  public ngOnInit() {
    this.userEntityService
      .getAllUserEntities()
      .subscribe(
        (userEntities) => {
          this.userEntities = userEntities;
          console.log(userEntities);
        }
      );
  }

  // add point to chart serie
  addPoint() {
    this.chart.addPoint(Math.floor(Math.random() * 10));
  }

  addUserEntity() {
    console.log(this.newUserEntity);
    this.userEntityService.addUserEntity(this.newUserEntity)
      .subscribe(
        (newUserEntity) => {
           this.userEntities = this.userEntities.concat(newUserEntity);
        }
      );

    //this.add.emit(this.newUserEntity);
    //this.newUserEntity = new UserEntity();
  }

  removeUserEntity(userEntity) {
    this.userEntityService
      .deleteUserEntityById(userEntity.id)
      .subscribe(
        (_) => {
          this.userEntities = this.userEntities.filter((ue) => ue.id !== userEntity.id);
        }
      );
  }

}
