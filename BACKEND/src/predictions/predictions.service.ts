import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

@Injectable()
export class PredictionsService {
  constructor(private readonly http: HttpService) {}

  async predictSingle(body: any): Promise<any> {
    try {
      const resp = await firstValueFrom(
        this.http.post(`${AI_ENGINE_URL}/predict/single`, body),
      );
      return resp.data;
    } catch (err) {
      return { error: 'AI Engine unavailable', detail: err?.message };
    }
  }

  async getFuzzyDetail(dataset: string, drug_idx: number, disease_idx: number): Promise<any> {
    try {
      const resp = await firstValueFrom(
        this.http.post(
          `${AI_ENGINE_URL}/predict/fuzzy_detail?dataset=${dataset}&drug_idx=${drug_idx}&disease_idx=${disease_idx}`,
          {},
        ),
      );
      return resp.data;
    } catch (err) {
      return { error: 'AI Engine unavailable', detail: err?.message };
    }
  }

  async predictMatrix(body: any): Promise<any> {
    try {
      const resp = await firstValueFrom(
        this.http.post(`${AI_ENGINE_URL}/predict/matrix`, body),
      );
      return resp.data;
    } catch (err) {
      return { error: 'AI Engine unavailable', detail: err?.message };
    }
  }

  async getTrainingResults(dataset: string, model: string): Promise<any> {
    try {
      const resp = await firstValueFrom(
        this.http.get(`${AI_ENGINE_URL}/results/training?dataset=${dataset}&model=${model}`),
      );
      return resp.data;
    } catch (err) {
      return { error: 'AI Engine unavailable', detail: err?.message };
    }
  }
}
