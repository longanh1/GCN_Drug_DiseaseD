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
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PredictionsController = void 0;
const common_1 = require("@nestjs/common");
const predictions_service_1 = require("./predictions.service");
let PredictionsController = class PredictionsController {
    constructor(predictionsService) {
        this.predictionsService = predictionsService;
    }
    async predictSingle(body) {
        return this.predictionsService.predictSingle(body);
    }
    async fuzzyDetail(dataset = 'C-dataset', drug_idx, disease_idx) {
        return this.predictionsService.getFuzzyDetail(dataset, +drug_idx, +disease_idx);
    }
    async predictMatrix(body) {
        return this.predictionsService.predictMatrix(body);
    }
    async getResults(dataset = 'C-dataset', model = 'AMNTDDA_Fuzzy') {
        return this.predictionsService.getTrainingResults(dataset, model);
    }
};
exports.PredictionsController = PredictionsController;
__decorate([
    (0, common_1.Post)('single'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object]),
    __metadata("design:returntype", Promise)
], PredictionsController.prototype, "predictSingle", null);
__decorate([
    (0, common_1.Post)('fuzzy-detail'),
    __param(0, (0, common_1.Query)('dataset')),
    __param(1, (0, common_1.Query)('drug_idx')),
    __param(2, (0, common_1.Query)('disease_idx')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, Number, Number]),
    __metadata("design:returntype", Promise)
], PredictionsController.prototype, "fuzzyDetail", null);
__decorate([
    (0, common_1.Post)('matrix'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object]),
    __metadata("design:returntype", Promise)
], PredictionsController.prototype, "predictMatrix", null);
__decorate([
    (0, common_1.Get)('results'),
    __param(0, (0, common_1.Query)('dataset')),
    __param(1, (0, common_1.Query)('model')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, Object]),
    __metadata("design:returntype", Promise)
], PredictionsController.prototype, "getResults", null);
exports.PredictionsController = PredictionsController = __decorate([
    (0, common_1.Controller)('predictions'),
    __metadata("design:paramtypes", [predictions_service_1.PredictionsService])
], PredictionsController);
//# sourceMappingURL=predictions.controller.js.map