import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { DrugsModule } from './drugs/drugs.module';
import { DiseasesModule } from './diseases/diseases.module';
import { ProteinsModule } from './proteins/proteins.module';
import { PredictionsModule } from './predictions/predictions.module';
import { ComparisonModule } from './comparison/comparison.module';
import { HistoryModule } from './history/history.module';
import { StatsModule } from './stats/stats.module';
import { AuthModule } from './auth/auth.module';
import { User } from './users/user.entity';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'postgres',
      database: process.env.DB_NAME || 'pharmalink',
      entities: [User],
      synchronize: true, // auto create tables in dev
      logging: false,
    }),
    HttpModule,
    AuthModule,
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
