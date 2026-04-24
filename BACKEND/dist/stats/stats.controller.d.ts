import { StatsService } from './stats.service';
export declare class StatsController {
    private readonly statsService;
    constructor(statsService: StatsService);
    getStats(dataset?: string): {
        dataset: string;
        num_drugs: number;
        num_diseases: number;
        num_proteins: number;
        num_known_links: number;
        num_models: number;
        best_auc: number;
        models: {
            name: string;
            status: string;
            summary: any;
        }[];
    };
    getGlobalStats(): {
        totalDrugs: number;
        totalDiseases: number;
        totalProteins: number;
        totalLinks: number;
        datasets: string[];
    };
    getDatasets(): {
        datasets: string[];
    };
}
