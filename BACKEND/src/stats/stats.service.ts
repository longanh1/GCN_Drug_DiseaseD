import { Injectable } from '@nestjs/common';
import { DrugsService } from '../drugs/drugs.service';
import { DiseasesService } from '../diseases/diseases.service';
import { ProteinsService } from '../proteins/proteins.service';
import { readCsv, listDatasets, readJson, aiDataPath, datasetPath } from '../shared/data.utils';

@Injectable()
export class StatsService {
  constructor(
    private drugsService: DrugsService,
    private diseasesService: DiseasesService,
    private proteinsService: ProteinsService,
  ) {}

  getDatasets(): string[] {
    return listDatasets();
  }

  getStats(dataset: string) {
    const numDrugs    = this.drugsService.countDrugs(dataset);
    const numDiseases = this.diseasesService.countDiseases(dataset);
    const numProteins = this.proteinsService.countProteins(dataset);

    // Count known links
    const assocRows  = readCsv(datasetPath(dataset, 'DrugDiseaseAssociationNumber.csv'));
    const knownLinks = assocRows.length;

    // Training results
    const fuzzySummary = readJson(aiDataPath('results', `${dataset}_AMNTDDA_Fuzzy_summary.json`));
    const gcnSummary   = readJson(aiDataPath('results', `${dataset}_AMNTDDA_summary.json`));

    const bestAuc = fuzzySummary?.AUC_mean ?? gcnSummary?.AUC_mean ?? null;

    return {
      dataset,
      num_drugs:      numDrugs,
      num_diseases:   numDiseases,
      num_proteins:   numProteins,
      num_known_links: knownLinks,
      num_models:     2,
      best_auc:       bestAuc ? Number(bestAuc.toFixed(4)) : null,
      models: [
        { name: 'GCN AMNTDDA',      status: 'active', summary: gcnSummary },
        { name: 'GCN + Fuzzy',      status: 'active', summary: fuzzySummary },
      ],
    };
  }

  getGlobalStats() {
    const datasets = this.getDatasets();
    let totalDrugs = 0, totalDiseases = 0, totalProteins = 0, totalLinks = 0;
    for (const ds of datasets) {
      try {
        totalDrugs    += this.drugsService.countDrugs(ds);
        totalDiseases += this.diseasesService.countDiseases(ds);
        totalProteins += this.proteinsService.countProteins(ds);
        const rows = readCsv(datasetPath(ds, 'DrugDiseaseAssociationNumber.csv'));
        totalLinks += rows.length;
      } catch {}
    }
    return { totalDrugs, totalDiseases, totalProteins, totalLinks, datasets };
  }
}
