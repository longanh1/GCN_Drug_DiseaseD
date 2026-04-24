import { Module } from '@nestjs/common';
import { ProteinsController } from './proteins.controller';
import { ProteinsService } from './proteins.service';

@Module({
  controllers: [ProteinsController],
  providers: [ProteinsService],
  exports: [ProteinsService],
})
export class ProteinsModule {}
