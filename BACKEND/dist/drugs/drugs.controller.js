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
exports.DrugsController = void 0;
const common_1 = require("@nestjs/common");
const drugs_service_1 = require("./drugs.service");
let DrugsController = class DrugsController {
    constructor(drugsService) {
        this.drugsService = drugsService;
    }
    getDrugs(dataset = 'C-dataset', search, limit = 200) {
        const drugs = this.drugsService.getDrugs(dataset, search, +limit);
        return { drugs, total: drugs.length, dataset };
    }
    getDrug(idx, dataset = 'C-dataset') {
        const drug = this.drugsService.getDrugByIdx(dataset, idx);
        if (!drug)
            return { error: 'Drug not found' };
        return drug;
    }
};
exports.DrugsController = DrugsController;
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, common_1.Query)('dataset')),
    __param(1, (0, common_1.Query)('search')),
    __param(2, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, String, Object]),
    __metadata("design:returntype", void 0)
], DrugsController.prototype, "getDrugs", null);
__decorate([
    (0, common_1.Get)(':idx'),
    __param(0, (0, common_1.Param)('idx', common_1.ParseIntPipe)),
    __param(1, (0, common_1.Query)('dataset')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Number, Object]),
    __metadata("design:returntype", void 0)
], DrugsController.prototype, "getDrug", null);
exports.DrugsController = DrugsController = __decorate([
    (0, common_1.Controller)('drugs'),
    __metadata("design:paramtypes", [drugs_service_1.DrugsService])
], DrugsController);
//# sourceMappingURL=drugs.controller.js.map