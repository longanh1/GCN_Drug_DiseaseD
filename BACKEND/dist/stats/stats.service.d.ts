import { DrugsService } from '../drugs/drugs.service';
import { DiseasesService } from '../diseases/diseases.service';
import { ProteinsService } from '../proteins/proteins.service';
export declare class StatsService {
    private drugsService;
    private diseasesService;
    private proteinsService;
    constructor(drugsService: DrugsService, diseasesService: DiseasesService, proteinsService: ProteinsService);
    getDatasets(): string[];
    getStats(dataset: string): {
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
}
