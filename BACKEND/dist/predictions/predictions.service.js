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
exports.PredictionsService = void 0;
const common_1 = require("@nestjs/common");
const axios_1 = require("@nestjs/axios");
const rxjs_1 = require("rxjs");
const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';
let PredictionsService = class PredictionsService {
    constructor(http) {
        this.http = http;
    }
    async predictSingle(body) {
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.post(`${AI_ENGINE_URL}/predict/single`, body));
            return resp.data;
        }
        catch (err) {
            return { error: 'AI Engine unavailable', detail: err?.message };
        }
    }
    async getFuzzyDetail(dataset, drug_idx, disease_idx) {
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.post(`${AI_ENGINE_URL}/predict/fuzzy_detail?dataset=${dataset}&drug_idx=${drug_idx}&disease_idx=${disease_idx}`, {}));
            return resp.data;
        }
        catch (err) {
            return { error: 'AI Engine unavailable', detail: err?.message };
        }
    }
    async predictMatrix(body) {
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.post(`${AI_ENGINE_URL}/predict/matrix`, body));
            return resp.data;
        }
        catch (err) {
            return { error: 'AI Engine unavailable', detail: err?.message };
        }
    }
    async getTrainingResults(dataset, model) {
        try {
            const resp = await (0, rxjs_1.firstValueFrom)(this.http.get(`${AI_ENGINE_URL}/results/training?dataset=${dataset}&model=${model}`));
            return resp.data;
        }
        catch (err) {
            return { error: 'AI Engine unavailable', detail: err?.message };
        }
    }
};
exports.PredictionsService = PredictionsService;
exports.PredictionsService = PredictionsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [axios_1.HttpService])
], PredictionsService);
//# sourceMappingURL=predictions.service.js.map