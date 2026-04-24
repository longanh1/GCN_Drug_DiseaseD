import { HttpService } from '@nestjs/axios';
export declare class ComparisonService {
    private readonly http;
    constructor(http: HttpService);
    getComparison(dataset: string): Promise<any>;
    compareMatrix(body: any): Promise<any>;
}
