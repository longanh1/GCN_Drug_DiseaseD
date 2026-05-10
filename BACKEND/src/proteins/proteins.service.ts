import { Injectable } from '@nestjs/common';
import { readCsv, datasetPath } from '../shared/data.utils';

@Injectable()
export class ProteinsService {
  private _cache = new Map<string, any[]>();

  getProteins(dataset: string, limit = 100): any[] {
    if (!this._cache.has(dataset)) {
      const rows = readCsv(datasetPath(dataset, 'ProteinInformation.csv'));
      this._cache.set(dataset, rows.map((r, i) => ({ idx: i, ...r })));
    }
    return this._cache.get(dataset)!.slice(0, limit);
  }

  countProteins(dataset: string): number {
    return this.getProteins(dataset, 99999).length;
  }
}
