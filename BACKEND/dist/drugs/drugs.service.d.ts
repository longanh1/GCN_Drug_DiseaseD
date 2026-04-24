export declare class DrugsService {
    private _cache;
    getDrugs(dataset: string, search?: string, limit?: number): any[];
    getDrugByIdx(dataset: string, idx: number): any | undefined;
    countDrugs(dataset: string): number;
}
