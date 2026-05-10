import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { DrugsModule } from './drugs/drugs.module';
import { DiseasesModule } from './diseases/diseases.module';
import { ProteinsModule } from './proteins/proteins.module';
import { PredictionsModule } from './predictions/predictions.module';
import { ComparisonModule } from './comparison/comparison.module';
import { HistoryModule } from './history/history.module';
import { StatsModule } from './stats/stats.module';

@Module({
  imports: [
    HttpModule,
    DrugsModule,
    DiseasesModule,
    ProteinsModule,
    PredictionsModule,
    ComparisonModule,
    HistoryModule,
    StatsModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
