export declare class HistoryService {
    private history;
    constructor();
    private _load;
    private _save;
    addEntry(entry: any): any;
    getAll(limit?: number): any[];
    clear(): {
        message: string;
    };
}
