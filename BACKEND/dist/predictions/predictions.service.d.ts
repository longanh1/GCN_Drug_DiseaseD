import { HttpService } from '@nestjs/axios';
export declare class PredictionsService {
    private readonly http;
    constructor(http: HttpService);
    predictSingle(body: any): Promise<any>;
    getFuzzyDetail(dataset: string, drug_idx: number, disease_idx: number): Promise<any>;
    predictMatrix(body: any): Promise<any>;
    getTrainingResults(dataset: string, model: string): Promise<any>;
}
