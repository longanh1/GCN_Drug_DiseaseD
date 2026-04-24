import { Injectable } from '@nestjs/common';
import { readCsvNoHeader, datasetPath } from '../shared/data.utils';
import * as fs from 'fs';

@Injectable()
export class DiseasesService {
  private _cache = new Map<string, any[]>();

  getDiseases(dataset: string, search?: string, limit = 200): any[] {
    if (!this._cache.has(dataset)) {
      const rows = readCsvNoHeader(datasetPath(dataset, 'DiseaseFeature.csv'));
      const diseases = rows.map((r, i) => ({
        idx: i,
        id: String(r[0]),
        name: `OMIM:${String(r[0])}`,
      }));
      this._cache.set(dataset, diseases);
    }
    let diseases = this._cache.get(dataset)!;
    if (search) {
      const sl = search.toLowerCase();
      diseases = diseases.filter(
        d => d.id.toLowerCase().includes(sl) || d.name.toLowerCase().includes(sl),
      );
    }
    return diseases.slice(0, limit);
  }

  getDiseaseByIdx(dataset: string, idx: number): any | undefined {
    return this.getDiseases(dataset, undefined, 99999)[idx];
  }

  countDiseases(dataset: string): number {
    return this.getDiseases(dataset, undefined, 99999).length;
  }
}
