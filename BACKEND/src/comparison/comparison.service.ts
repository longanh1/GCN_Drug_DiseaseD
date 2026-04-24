import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { readJson, aiDataPath } from '../shared/data.utils';

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

@Injectable()
export class ComparisonService {
  constructor(private readonly http: HttpService) {}

  async getComparison(dataset: string): Promise<any> {
    // Try local file first (fast), fall back to AI Engine
    const local = readJson(aiDataPath('results', `${dataset}_comparison.json`));
    if (local) return local;

    try {
      const resp = await firstValueFrom(
        this.http.get(`${AI_ENGINE_URL}/results/comparison?dataset=${dataset}`),
      );
      return resp.data;
    } catch {
      return { dataset, models: {} };
    }
  }

  async compareMatrix(body: any): Promise<any> {
    try {
      const resp = await firstValueFrom(
        this.http.post(`${AI_ENGINE_URL}/predict/matrix`, {
          ...body,
          model: 'AMNTDDA_Fuzzy',
        }),
      );
      const fuzzyResult = resp.data;

      const resp2 = await firstValueFrom(
        this.http.post(`${AI_ENGINE_URL}/predict/matrix`, {
          ...body,
          model: 'AMNTDDA',
        }),
      );
      const gcnResult = resp2.data;

      // Merge cells
      const merged = (fuzzyResult.cells || []).map((cell: any, i: number) => {
        const gcnCell = (gcnResult.cells || [])[i] || {};
        return {
          ...cell,
          gcn_score:   gcnCell.gcn_score ?? cell.gcn_score,
          fuzzy_score: cell.fuzzy_score,
          delta:       Number(((cell.fuzzy_score ?? 0) - (gcnCell.gcn_score ?? 0)).toFixed(4)),
        };
      });

      return {
        dataset:   body.dataset,
        cells:     merged,
        gcn_avg:   merged.length ? merged.reduce((s, c) => s + c.gcn_score, 0) / merged.length : 0,
        fuzzy_avg: merged.length ? merged.reduce((s, c) => s + c.fuzzy_score, 0) / merged.length : 0,
      };
    } catch (err) {
      return { error: 'AI Engine unavailable', detail: err?.message };
    }
  }
}
