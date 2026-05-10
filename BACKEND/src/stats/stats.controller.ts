import { Controller, Get, Query } from '@nestjs/common';
import { StatsService } from './stats.service';

@Controller('stats')
export class StatsController {
  constructor(private readonly statsService: StatsService) {}

  @Get()
  getStats(@Query('dataset') dataset = 'C-dataset') {
    return this.statsService.getStats(dataset);
  }

  @Get('global')
  getGlobalStats() {
    return this.statsService.getGlobalStats();
  }

  @Get('datasets')
  getDatasets() {
    return { datasets: this.statsService.getDatasets() };
  }
}
