import { ProteinsService } from './proteins.service';
export declare class ProteinsController {
    private readonly proteinsService;
    constructor(proteinsService: ProteinsService);
    getProteins(dataset?: string, limit?: number): {
        proteins: any[];
        total: number;
        dataset: string;
    };
}
