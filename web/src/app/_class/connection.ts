export class Connection {
  id: number;
  key: string = '';
  first_name: string = '';
  last_name: string = '';
  occupation: string = '';
  public_identifier: string = '';
  entity_urn: string = '';
  connected_at: Date = new Date();

  constructor(values: Object = {}) {
    Object.assign(this, values);
  }
}
