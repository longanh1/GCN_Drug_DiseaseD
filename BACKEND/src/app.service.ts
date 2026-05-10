import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello(): string {
    return 'PharmaLink GCN Drug-Disease Platform';
  }
}
