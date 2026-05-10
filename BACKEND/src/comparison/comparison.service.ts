import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { readJson, aiDataPath } from '../shared/data.utils';

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

const ABLATION_VARIANT_ORDER = [
  'sim_only', 'gcn_only',
  'sim_transformer', 'gcn_transformer',
  'sim_gcn', 'sim_transformer_gcn',
  'full',
] as const;

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

  async getAblationComparison(dataset: string): Promise<any> {
    // Try local file first
    const local = readJson(aiDataPath('results', `${dataset}_ablation_comparison.json`));
    if (local) return local;

    try {
      const resp = await firstValueFrom(
        this.http.get(`${AI_ENGINE_URL}/results/ablation?dataset=${dataset}`),
      );
      return resp.data;
    } catch {
      return { dataset, trained: false, variant_order: ABLATION_VARIANT_ORDER, variants: {} };
    }
  }

  async getAblationAllVariants(dataset: string): Promise<any> {
    // Try local file first
    const local = readJson(aiDataPath('results', `${dataset}_ablation_comparison.json`));
    if (local) {
      return {
        dataset,
        variant_order: ABLATION_VARIANT_ORDER,
        variants: local.variants ?? {},
      };
    }

    try {
      const resp = await firstValueFrom(
        this.http.get(`${AI_ENGINE_URL}/results/ablation/all_variants?dataset=${dataset}`),
      );
      return resp.data;
    } catch {
      return { dataset, variant_order: ABLATION_VARIANT_ORDER, variants: {} };
    }
  }

  async getAblationVariant(dataset: string, variant: string): Promise<any> {
    // Try local per-variant summary
    const summary = readJson(
      aiDataPath('results', `${dataset}_ablation_${variant}_summary.json`)
    );

    try {
      const resp = await firstValueFrom(
        this.http.get(
          `${AI_ENGINE_URL}/results/ablation/variant?dataset=${dataset}&variant=${variant}`
        ),
      );
      return resp.data;
    } catch {
      return { dataset, variant, trained: !!summary, summary: summary ?? null, folds: [] };
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
