import { Controller, Get, Post, Delete, Body, Query } from '@nestjs/common';
import { HistoryService } from './history.service';

@Controller('history')
export class HistoryController {
  constructor(private readonly historyService: HistoryService) {}

  @Get()
  getHistory(@Query('limit') limit = 50) {
    return { history: this.historyService.getAll(+limit) };
  }

  @Post()
  addEntry(@Body() body: any) {
    return this.historyService.addEntry(body);
  }

  @Delete()
  clearHistory() {
    return this.historyService.clear();
  }
}
