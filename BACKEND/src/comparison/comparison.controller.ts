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

  /** GET /comparison/ablation?dataset=C-dataset
   *  Returns the full ablation comparison JSON (all 7 variants). */
  @Get('ablation')
  async getAblationComparison(@Query('dataset') dataset = 'C-dataset') {
    return this.comparisonService.getAblationComparison(dataset);
  }

  /** GET /comparison/ablation/variants?dataset=C-dataset
   *  Returns summary metrics for every variant (no fold details). */
  @Get('ablation/variants')
  async getAblationAllVariants(@Query('dataset') dataset = 'C-dataset') {
    return this.comparisonService.getAblationAllVariants(dataset);
  }

  /** GET /comparison/ablation/variant?dataset=C-dataset&variant=full
   *  Returns summary + per-fold results for one variant. */
  @Get('ablation/variant')
  async getAblationVariant(
    @Query('dataset') dataset = 'C-dataset',
    @Query('variant') variant = 'full',
  ) {
    return this.comparisonService.getAblationVariant(dataset, variant);
  }
}
