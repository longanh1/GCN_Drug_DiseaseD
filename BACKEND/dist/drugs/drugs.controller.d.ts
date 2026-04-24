import { DrugsService } from './drugs.service';
export declare class DrugsController {
    private readonly drugsService;
    constructor(drugsService: DrugsService);
    getDrugs(dataset?: string, search?: string, limit?: number): {
        drugs: any[];
        total: number;
        dataset: string;
    };
    getDrug(idx: number, dataset?: string): any;
}
