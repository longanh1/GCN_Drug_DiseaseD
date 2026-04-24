import { HistoryService } from './history.service';
export declare class HistoryController {
    private readonly historyService;
    constructor(historyService: HistoryService);
    getHistory(limit?: number): {
        history: any[];
    };
    addEntry(body: any): any;
    clearHistory(): {
        message: string;
    };
}
