import { Injectable } from '@angular/core';
import { Measurement } from '../_class/measurement';
import { ApiService } from './api.service';
import { Observable } from 'rxjs/Observable';

@Injectable()
export class MeasurementService {

  constructor(
    private api: ApiService
  ) {
  }

  // Simulate POST /userentity
  addUserEntity(measurement: Measurement): Observable<Measurement> {
    return this.api.createMeasurement(measurement);
  }

  // Simulate DELETE /userentity/:id
  deleteUserEntityById(measurementId: number): Observable<Measurement> {
    return this.api.deleteUserEntityById(measurementId);
  }

  // Simulate PUT /userentity/:id
  updateMeasurement(measurement: Measurement): Observable<Measurement> {
    return this.api.updateMeasurement(measurement);
  }

  // Simulate GET /userentity
  getAllMeasurements(): Observable<Measurement[]> {
    return this.api.getAllMeasurements();
  }

  // Simulate GET /userentity/:id
  getUserEntityById(measurementId: number): Observable<Measurement> {
    return this.api.getMeasurementById(measurementId);
  }
}