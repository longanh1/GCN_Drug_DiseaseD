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
exports.ProteinsController = void 0;
const common_1 = require("@nestjs/common");
const proteins_service_1 = require("./proteins.service");
let ProteinsController = class ProteinsController {
    constructor(proteinsService) {
        this.proteinsService = proteinsService;
    }
    getProteins(dataset = 'C-dataset', limit = 100) {
        const proteins = this.proteinsService.getProteins(dataset, +limit);
        return { proteins, total: proteins.length, dataset };
    }
};
exports.ProteinsController = ProteinsController;
__decorate([
    (0, common_1.Get)(),
    __param(0, (0, common_1.Query)('dataset')),
    __param(1, (0, common_1.Query)('limit')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object, Object]),
    __metadata("design:returntype", void 0)
], ProteinsController.prototype, "getProteins", null);
exports.ProteinsController = ProteinsController = __decorate([
    (0, common_1.Controller)('proteins'),
    __metadata("design:paramtypes", [proteins_service_1.ProteinsService])
], ProteinsController);
//# sourceMappingURL=proteins.controller.js.map