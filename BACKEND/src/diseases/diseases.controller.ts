import { Controller, Get, Query, Param, ParseIntPipe } from '@nestjs/common';
import { DiseasesService } from './diseases.service';

@Controller('diseases')
export class DiseasesController {
  constructor(private readonly diseasesService: DiseasesService) {}

  @Get()
  getDiseases(
    @Query('dataset') dataset = 'C-dataset',
    @Query('search') search?: string,
    @Query('limit') limit = 200,
  ) {
    const diseases = this.diseasesService.getDiseases(dataset, search, +limit);
    return { diseases, total: diseases.length, dataset };
  }

  @Get(':idx')
  getDisease(
    @Param('idx', ParseIntPipe) idx: number,
    @Query('dataset') dataset = 'C-dataset',
  ) {
    const dis = this.diseasesService.getDiseaseByIdx(dataset, idx);
    if (!dis) return { error: 'Disease not found' };
    return dis;
  }
}
