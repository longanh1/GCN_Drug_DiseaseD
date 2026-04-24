import { Controller, Post, Get, Body, Query } from '@nestjs/common';
import { PredictionsService } from './predictions.service';

@Controller('predictions')
export class PredictionsController {
  constructor(private readonly predictionsService: PredictionsService) {}

  @Post('single')
  async predictSingle(@Body() body: any) {
    return this.predictionsService.predictSingle(body);
  }

  @Post('fuzzy-detail')
  async fuzzyDetail(
    @Query('dataset') dataset = 'C-dataset',
    @Query('drug_idx') drug_idx: number,
    @Query('disease_idx') disease_idx: number,
  ) {
    return this.predictionsService.getFuzzyDetail(dataset, +drug_idx, +disease_idx);
  }

  @Post('matrix')
  async predictMatrix(@Body() body: any) {
    return this.predictionsService.predictMatrix(body);
  }

  @Get('results')
  async getResults(
    @Query('dataset') dataset = 'C-dataset',
    @Query('model') model = 'AMNTDDA_Fuzzy',
  ) {
    return this.predictionsService.getTrainingResults(dataset, model);
  }
}
