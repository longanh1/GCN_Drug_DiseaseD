"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProteinsService = void 0;
const common_1 = require("@nestjs/common");
const data_utils_1 = require("../shared/data.utils");
let ProteinsService = class ProteinsService {
    constructor() {
        this._cache = new Map();
    }
    getProteins(dataset, limit = 100) {
        if (!this._cache.has(dataset)) {
            const rows = (0, data_utils_1.readCsv)((0, data_utils_1.datasetPath)(dataset, 'ProteinInformation.csv'));
            this._cache.set(dataset, rows.map((r, i) => ({ idx: i, ...r })));
        }
        return this._cache.get(dataset).slice(0, limit);
    }
    countProteins(dataset) {
        return this.getProteins(dataset, 99999).length;
    }
};
exports.ProteinsService = ProteinsService;
exports.ProteinsService = ProteinsService = __decorate([
    (0, common_1.Injectable)()
], ProteinsService);
//# sourceMappingURL=proteins.service.js.map