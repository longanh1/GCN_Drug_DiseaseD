import { ComparisonService } from './comparison.service';
export declare class ComparisonController {
    private readonly comparisonService;
    constructor(comparisonService: ComparisonService);
    getComparison(dataset?: string): Promise<any>;
    compareMatrix(body: any): Promise<any>;
}
