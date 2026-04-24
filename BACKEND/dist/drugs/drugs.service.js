"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DrugsService = void 0;
const common_1 = require("@nestjs/common");
const data_utils_1 = require("../shared/data.utils");
let DrugsService = class DrugsService {
    constructor() {
        this._cache = new Map();
    }
    getDrugs(dataset, search, limit = 200) {
        if (!this._cache.has(dataset)) {
            const rows = (0, data_utils_1.readCsv)((0, data_utils_1.datasetPath)(dataset, 'DrugInformation.csv'));
            this._cache.set(dataset, rows.map((r, i) => ({ idx: i, ...r })));
        }
        let drugs = this._cache.get(dataset);
        if (search) {
            const sl = search.toLowerCase();
            drugs = drugs.filter(d => String(d.name || '').toLowerCase().includes(sl) ||
                String(d.id || '').toLowerCase().includes(sl));
        }
        return drugs.slice(0, limit);
    }
    getDrugByIdx(dataset, idx) {
        return this.getDrugs(dataset)[idx];
    }
    countDrugs(dataset) {
        return this.getDrugs(dataset, undefined, 99999).length;
    }
};
exports.DrugsService = DrugsService;
exports.DrugsService = DrugsService = __decorate([
    (0, common_1.Injectable)()
], DrugsService);
//# sourceMappingURL=drugs.service.js.map