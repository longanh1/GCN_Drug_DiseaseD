"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ComparisonService = void 0;
const common_1 = require("@nestjs/common");
const axios_1 = require("@nestjs/axios");
const rxjs_1 = require("rxjs");
const data_utils_1 = require("../shared/data.utils");
const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';
let ComparisonService = class ComparisonService {
    constructor(http) {
        this.http = http;
    }
    async getComparison(dataset) {
        const local = (0, data_utils_1.readJson)((0, data_utils_1.aiDataPath)('results', `${dataset}_comparison.json`));
        if (local)
            return local;
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.get(`${AI_ENGINE_URL}/results/comparison?dataset=${dataset}`));
            return resp.data;
        }
        catch {
            return { dataset, models: {} };
        }
    }
    async compareMatrix(body) {
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.post(`${AI_ENGINE_URL}/predict/matrix`, {
                ...body,
                model: 'AMNTDDA_Fuzzy',
            }));
            const fuzzyResult = resp.data;
            const resp2 = await (0, rxjs_1.firstValueFrom)(this.http.post(`${AI_ENGINE_URL}/predict/matrix`, {
                ...body,
                model: 'AMNTDDA',
            }));
            const gcnResult = resp2.data;
            const merged = (fuzzyResult.cells || []).map((cell, i) => {
                const gcnCell = (gcnResult.cells || [])[i] || {};
                return {
                    ...cell,
                    gcn_score: gcnCell.gcn_score ?? cell.gcn_score,
                    fuzzy_score: cell.fuzzy_score,
                    delta: Number(((cell.fuzzy_score ?? 0) - (gcnCell.gcn_score ?? 0)).toFixed(4)),
                };
            });
            return {
                dataset: body.dataset,
                cells: merged,
                gcn_avg: merged.length ? merged.reduce((s, c) => s + c.gcn_score, 0) / merged.length : 0,
                fuzzy_avg: merged.length ? merged.reduce((s, c) => s + c.fuzzy_score, 0) / merged.length : 0,
            };
        }
        catch (err) {
            return { error: 'AI Engine unavailable', detail: err?.message };
        }
    }
};
exports.ComparisonService = ComparisonService;
exports.ComparisonService = ComparisonService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [axios_1.HttpService])
], ComparisonService);
//# sourceMappingURL=comparison.service.js.map