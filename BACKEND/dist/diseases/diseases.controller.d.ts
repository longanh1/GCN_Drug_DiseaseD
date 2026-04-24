import { DiseasesService } from './diseases.service';
export declare class DiseasesController {
    private readonly diseasesService;
    constructor(diseasesService: DiseasesService);
    getDiseases(dataset?: string, search?: string, limit?: number): {
        diseases: any[];
        total: number;
        dataset: string;
    };
    getDisease(idx: number, dataset?: string): any;
}
