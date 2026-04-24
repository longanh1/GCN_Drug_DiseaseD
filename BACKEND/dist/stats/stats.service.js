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
exports.StatsService = void 0;
const common_1 = require("@nestjs/common");
const drugs_service_1 = require("../drugs/drugs.service");
const diseases_service_1 = require("../diseases/diseases.service");
const proteins_service_1 = require("../proteins/proteins.service");
const data_utils_1 = require("../shared/data.utils");
let StatsService = class StatsService {
    constructor(drugsService, diseasesService, proteinsService) {
        this.drugsService = drugsService;
        this.diseasesService = diseasesService;
        this.proteinsService = proteinsService;
    }
    getDatasets() {
        return (0, data_utils_1.listDatasets)();
    }
    getStats(dataset) {
        const numDrugs = this.drugsService.countDrugs(dataset);
        const numDiseases = this.diseasesService.countDiseases(dataset);
        const numProteins = this.proteinsService.countProteins(dataset);
        const assocRows = (0, data_utils_1.readCsv)((0, data_utils_1.datasetPath)(dataset, 'DrugDiseaseAssociationNumber.csv'));
        const knownLinks = assocRows.length;
        const fuzzySummary = (0, data_utils_1.readJson)((0, data_utils_1.aiDataPath)('results', `${dataset}_AMNTDDA_Fuzzy_summary.json`));
        const gcnSummary = (0, data_utils_1.readJson)((0, data_utils_1.aiDataPath)('results', `${dataset}_AMNTDDA_summary.json`));
        const bestAuc = fuzzySummary?.AUC_mean ?? gcnSummary?.AUC_mean ?? null;
        return {
            dataset,
            num_drugs: numDrugs,
            num_diseases: numDiseases,
            num_proteins: numProteins,
            num_known_links: knownLinks,
            num_models: 2,
            best_auc: bestAuc ? Number(bestAuc.toFixed(4)) : null,
            models: [
                { name: 'GCN AMNTDDA', status: 'active', summary: gcnSummary },
                { name: 'GCN + Fuzzy', status: 'active', summary: fuzzySummary },
            ],
        };
    }
    getGlobalStats() {
        const datasets = this.getDatasets();
        let totalDrugs = 0, totalDiseases = 0, totalProteins = 0, totalLinks = 0;
        for (const ds of datasets) {
            try {
                totalDrugs += this.drugsService.countDrugs(ds);
                totalDiseases += this.diseasesService.countDiseases(ds);
                totalProteins += this.proteinsService.countProteins(ds);
                const rows = (0, data_utils_1.readCsv)((0, data_utils_1.datasetPath)(ds, 'DrugDiseaseAssociationNumber.csv'));
                totalLinks += rows.length;
            }
            catch { }
        }
        return { totalDrugs, totalDiseases, totalProteins, totalLinks, datasets };
    }
};
exports.StatsService = StatsService;
exports.StatsService = StatsService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [drugs_service_1.DrugsService,
        diseases_service_1.DiseasesService,
        proteins_service_1.ProteinsService])
], StatsService);
//# sourceMappingURL=stats.service.js.map