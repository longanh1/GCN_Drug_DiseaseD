import { Controller, Get, Post, Query, Body } from '@nestjs/common';
import { ComparisonService } from './comparison.service';

@Controller('comparison')
export class ComparisonController {
  constructor(private readonly comparisonService: ComparisonService) {}

  @Get()
  async getComparison(@Query('dataset') dataset = 'C-dataset') {
    return this.comparisonService.getComparison(dataset);
  }

  @Post('matrix')
  async compareMatrix(@Body() body: any) {
    return this.comparisonService.compareMatrix(body);
  }
}
