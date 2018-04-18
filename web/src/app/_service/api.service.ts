import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { Http, Response, Headers, RequestOptions } from '@angular/http';
import { Connection } from '../_class/connection';
import { UserEntity } from '../_class/user-entity';
import { Measurement } from '../_class/measurement';
import { Observable } from 'rxjs/Observable';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/catch';
import 'rxjs/add/observable/throw';
//import 'rxjs/Rx';

const API_URL = environment.apiUrl;

@Injectable()
export class ApiService {

  constructor(
    private http: Http
  ) {
  }

  public getProfileConnections(profileId: number): Observable<Connection[]> {
    let headers: Headers = new Headers();
    headers.append("Authorization", "Token 94629c425a5fd26dc172e3916e168975b648d80c");

    let options = new RequestOptions({headers: headers});
    return this.http
      .get(API_URL + 'profiles/' + profileId + '/pconnections/', options)
      .map(response => {
        const connections = response.json();
        return connections.map((profile) => new Connection(profile));
      })
      .catch(this.handleError);
  }

  public getAllUserEntities(): Observable<UserEntity[]> {
    return this.http
      .get(API_URL + '/userentity/')
      .map(response => {
        const userEntities = response.json();
        return userEntities.results.map((userEntity) => new UserEntity(userEntity));
      })
      .catch(this.handleError);
  }

  public createUserEntity(userEntity: UserEntity): Observable<UserEntity> {
    return this.http
      .post(API_URL + '/userentity/', userEntity)
      .map(response => {
        return new UserEntity(response.json());
      })
      .catch(this.handleError);
  }

  public getUserEntityById(userEntityId: number): Observable<UserEntity> {
    return this.http
      .get(API_URL + '/userentity/' + userEntityId + '/')
      .map(response => {
        return new UserEntity(response.json());
      })
      .catch(this.handleError);
  }

  public updateUserEntity(userEntity: UserEntity): Observable<UserEntity> {
    return this.http
      .put(API_URL + '/userentity/' + userEntity.id + '/', userEntity)
      .map(response => {
        return new UserEntity(response.json());
      })
      .catch(this.handleError);
  }

  public deleteUserEntityById(userEntityId: number): Observable<null> {
    return this.http
      .delete(API_URL + '/userentity/' + userEntityId + '/')
      .map(response => null)
      .catch(this.handleError);
  }


  /* MEASUREMENT */

  public getAllMeasurements(): Observable<Measurement[]> {
    return this.http
      .get(API_URL + '/measurement/')
      .map(response => {
        const measurements = response.json();
        return measurements.results.map((measurement) => new Measurement(measurement));
      })
      .catch(this.handleError);
  }

  public createMeasurement(measurement: Measurement): Observable<Measurement> {
    return this.http
      .post(API_URL + '/measurement/', measurement)
      .map(response => {
        return new Measurement(response.json());
      })
      .catch(this.handleError);
  }

  public getMeasurementById(measurementId: number): Observable<Measurement> {
    return this.http
      .get(API_URL + '/measurement/' + measurementId + '/')
      .map(response => {
        return new Measurement(response.json());
      })
      .catch(this.handleError);
  }

  public updateMeasurement(measurement: Measurement): Observable<Measurement> {
    return this.http
      .put(API_URL + '/measurement/' + measurement.id + '/', measurement)
      .map(response => {
        return new Measurement(response.json());
      })
      .catch(this.handleError);
  }

  public deleteMeasurementById(measurementId: number): Observable<null> {
    return this.http
      .delete(API_URL + '/measurement/' + measurementId + '/')
      .map(response => null)
      .catch(this.handleError);
  }

  private handleError(error: Response | any) {
    console.error('ApiService::handleError', error);
    return Observable.throw(error);
  }
}
