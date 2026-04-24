import { Module } from '@nestjs/common';
import { StatsController } from './stats.controller';
import { StatsService } from './stats.service';
import { DrugsModule } from '../drugs/drugs.module';
import { DiseasesModule } from '../diseases/diseases.module';
import { ProteinsModule } from '../proteins/proteins.module';

@Module({
  imports: [DrugsModule, DiseasesModule, ProteinsModule],
  controllers: [StatsController],
  providers: [StatsService],
})
export class StatsModule {}
