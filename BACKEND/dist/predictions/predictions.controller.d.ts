import { PredictionsService } from './predictions.service';
export declare class PredictionsController {
    private readonly predictionsService;
    constructor(predictionsService: PredictionsService);
    predictSingle(body: any): Promise<any>;
    fuzzyDetail(dataset: string, drug_idx: number, disease_idx: number): Promise<any>;
    predictMatrix(body: any): Promise<any>;
    getResults(dataset?: string, model?: string): Promise<any>;
}
