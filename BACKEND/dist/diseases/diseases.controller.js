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
exports.DiseasesController = void 0;
const common_1 = require("@nestjs/common");
const diseases_service_1 = require("./diseases.service");
let DiseasesController = class DiseasesController {
    constructor(diseasesService) {
        this.diseasesService = diseasesService;
    }
    getDiseases(dataset = 'C-dataset', search, limit = 200) {
        const diseases = this.diseasesService.getDiseases(dataset, search, +limit);
        return { diseases, total: diseases.length, dataset };
    }
    getDisease(idx, dataset = 'C-dataset') {
        const dis = this.diseasesService.getDiseaseByIdx(dataset, idx);
        if (!dis)
            return { error: 'Disease not found' };
        return dis;
    }
};
exports.DiseasesController = DiseasesController;
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, common_1.Query)('dataset')),
    __param(1, (0, common_1.Query)('search')),
    __param(2, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, String, Object]),
    __metadata("design:returntype", void 0)
], DiseasesController.prototype, "getDiseases", null);
__decorate([
    (0, common_1.Get)(':idx'),
    __param(0, (0, common_1.Param)('idx', common_1.ParseIntPipe)),
    __param(1, (0, common_1.Query)('dataset')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Number, Object]),
    __metadata("design:returntype", void 0)
], DiseasesController.prototype, "getDisease", null);
exports.DiseasesController = DiseasesController = __decorate([
    (0, common_1.Controller)('diseases'),
    __metadata("design:paramtypes", [diseases_service_1.DiseasesService])
], DiseasesController);
//# sourceMappingURL=diseases.controller.js.map