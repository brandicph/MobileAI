export class Measurement {
  id: number;
  key: string = '';
  value: string = '';
  unit: string = '';
  user_entity: string = '';
  created_at: Date = new Date();
  updated_at: Date = new Date();

  constructor(values: Object = {}) {
    Object.assign(this, values);
  }
}