import { Injectable } from '@angular/core';
import { UserEntity } from '../_class/user-entity';
import { ApiService } from './api.service';
import { Observable } from 'rxjs/Observable';

@Injectable()
export class UserEntityService {

  constructor(
    private api: ApiService
  ) {
  }

  // Simulate POST /userentity
  addUserEntity(userEntity: UserEntity): Observable<UserEntity> {
    return this.api.createUserEntity(userEntity);
  }

  // Simulate DELETE /userentity/:id
  deleteUserEntityById(userEntityId: number): Observable<UserEntity> {
    return this.api.deleteUserEntityById(userEntityId);
  }

  // Simulate PUT /userentity/:id
  updateUserEntity(userEntity: UserEntity): Observable<UserEntity> {
    return this.api.updateUserEntity(userEntity);
  }

  // Simulate GET /userentity
  getAllUserEntities(): Observable<UserEntity[]> {
    return this.api.getAllUserEntities();
  }

  // Simulate GET /userentity/:id
  getUserEntityById(userEntityId: number): Observable<UserEntity> {
    return this.api.getUserEntityById(userEntityId);
  }

}