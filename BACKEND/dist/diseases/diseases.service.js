"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiseasesService = void 0;
const common_1 = require("@nestjs/common");
const data_utils_1 = require("../shared/data.utils");
let DiseasesService = class DiseasesService {
    constructor() {
        this._cache = new Map();
    }
    getDiseases(dataset, search, limit = 200) {
        if (!this._cache.has(dataset)) {
            const rows = (0, data_utils_1.readCsvNoHeader)((0, data_utils_1.datasetPath)(dataset, 'DiseaseFeature.csv'));
            const diseases = rows.map((r, i) => ({
                idx: i,
                id: String(r[0]),
                name: `OMIM:${String(r[0])}`,
            }));
            this._cache.set(dataset, diseases);
        }
        let diseases = this._cache.get(dataset);
        if (search) {
            const sl = search.toLowerCase();
            diseases = diseases.filter(d => d.id.toLowerCase().includes(sl) || d.name.toLowerCase().includes(sl));
        }
        return diseases.slice(0, limit);
    }
    getDiseaseByIdx(dataset, idx) {
        return this.getDiseases(dataset, undefined, 99999)[idx];
    }
    countDiseases(dataset) {
        return this.getDiseases(dataset, undefined, 99999).length;
    }
};
exports.DiseasesService = DiseasesService;
exports.DiseasesService = DiseasesService = __decorate([
    (0, common_1.Injectable)()
], DiseasesService);
//# sourceMappingURL=diseases.service.js.map