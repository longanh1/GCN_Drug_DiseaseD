"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.datasetPath = datasetPath;
exports.aiDataPath = aiDataPath;
exports.readCsv = readCsv;
exports.readCsvNoHeader = readCsvNoHeader;
exports.listDatasets = listDatasets;
exports.readJson = readJson;
const fs = require("fs");
const path = require("path");
const Papa = require("papaparse");
const AMDGT_DATA = path.resolve(__dirname, '../../../AMDGT_main/data');
const AI_DATA = path.resolve(__dirname, '../../../AI_ENGINE/data');
function datasetPath(dataset, file) {
    return path.join(AMDGT_DATA, dataset, file);
}
function aiDataPath(...parts) {
    return path.join(AI_DATA, ...parts);
}
function readCsv(filePath) {
    if (!fs.existsSync(filePath))
        return [];
    const content = fs.readFileSync(filePath, 'utf-8');
    const result = Papa.parse(content, { header: true, skipEmptyLines: true });
    return result.data;
}
function readCsvNoHeader(filePath) {
    if (!fs.existsSync(filePath))
        return [];
    const content = fs.readFileSync(filePath, 'utf-8');
    const result = Papa.parse(content, { header: false, skipEmptyLines: true });
    return result.data;
}
function listDatasets() {
    if (!fs.existsSync(AMDGT_DATA))
        return [];
    return fs.readdirSync(AMDGT_DATA).filter(d => fs.statSync(path.join(AMDGT_DATA, d)).isDirectory());
}
function readJson(filePath) {
    if (!fs.existsSync(filePath))
        return null;
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
    catch {
        return null;
    }
}
//# sourceMappingURL=data.utils.js.map