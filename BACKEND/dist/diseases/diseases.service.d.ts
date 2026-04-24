export declare class DiseasesService {
    private _cache;
    getDiseases(dataset: string, search?: string, limit?: number): any[];
    getDiseaseByIdx(dataset: string, idx: number): any | undefined;
    countDiseases(dataset: string): number;
}
