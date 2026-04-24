import { Injectable } from '@nestjs/common';
import { readCsv, readCsvNoHeader, datasetPath } from '../shared/data.utils';

@Injectable()
export class DrugsService {
  private _cache = new Map<string, any[]>();

  getDrugs(dataset: string, search?: string, limit = 200): any[] {
    if (!this._cache.has(dataset)) {
      const rows = readCsv(datasetPath(dataset, 'DrugInformation.csv'));
      this._cache.set(dataset, rows.map((r, i) => ({ idx: i, ...r })));
    }
    let drugs = this._cache.get(dataset)!;
    if (search) {
      const sl = search.toLowerCase();
      drugs = drugs.filter(
        d => String(d.name || '').toLowerCase().includes(sl) ||
             String(d.id || '').toLowerCase().includes(sl),
      );
    }
    return drugs.slice(0, limit);
  }

  getDrugByIdx(dataset: string, idx: number): any | undefined {
    return this.getDrugs(dataset)[idx];
  }

  countDrugs(dataset: string): number {
    return this.getDrugs(dataset, undefined, 99999).length;
  }
}
