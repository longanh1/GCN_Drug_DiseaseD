import { Controller, Get, Query } from '@nestjs/common';
import { ProteinsService } from './proteins.service';

@Controller('proteins')
export class ProteinsController {
  constructor(private readonly proteinsService: ProteinsService) {}

  @Get()
  getProteins(
    @Query('dataset') dataset = 'C-dataset',
    @Query('limit') limit = 100,
  ) {
    const proteins = this.proteinsService.getProteins(dataset, +limit);
    return { proteins, total: proteins.length, dataset };
  }
}
