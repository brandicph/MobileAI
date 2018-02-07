export class UserEntity {
  id: number;
  name: string = '';
  IMEI: string = '';
  IMSI: string = '';
  created_at: Date = new Date();
  updated_at: Date = new Date();

  constructor(values: Object = {}) {
    Object.assign(this, values);
  }
}