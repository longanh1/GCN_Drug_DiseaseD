import { Controller, Get, Query, Param, ParseIntPipe } from '@nestjs/common';
import { DrugsService } from './drugs.service';

@Controller('drugs')
export class DrugsController {
  constructor(private readonly drugsService: DrugsService) {}

  @Get()
  getDrugs(
    @Query('dataset') dataset = 'C-dataset',
    @Query('search') search?: string,
    @Query('limit') limit = 200,
  ) {
    const drugs = this.drugsService.getDrugs(dataset, search, +limit);
    return { drugs, total: drugs.length, dataset };
  }

  @Get(':idx')
  getDrug(
    @Param('idx', ParseIntPipe) idx: number,
    @Query('dataset') dataset = 'C-dataset',
  ) {
    const drug = this.drugsService.getDrugByIdx(dataset, idx);
    if (!drug) return { error: 'Drug not found' };
    return drug;
  }
}
