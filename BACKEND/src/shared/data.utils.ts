/**
 * Shared data-reading utilities for AMDGT_main CSV files.
 */
import * as fs from 'fs';
import * as path from 'path';
import * as Papa from 'papaparse';

const AMDGT_DATA = path.resolve(__dirname, '../../../AMDGT_main/data');
const AI_DATA    = path.resolve(__dirname, '../../../AI_ENGINE/data');

export function datasetPath(dataset: string, file: string): string {
  return path.join(AMDGT_DATA, dataset, file);
}

export function aiDataPath(...parts: string[]): string {
  return path.join(AI_DATA, ...parts);
}

export function readCsv(filePath: string): any[] {
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, 'utf-8');
  const result = Papa.parse(content, { header: true, skipEmptyLines: true });
  return result.data as any[];
}

export function readCsvNoHeader(filePath: string): string[][] {
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, 'utf-8');
  const result = Papa.parse(content, { header: false, skipEmptyLines: true });
  return result.data as string[][];
}

export function listDatasets(): string[] {
  if (!fs.existsSync(AMDGT_DATA)) return [];
  return fs.readdirSync(AMDGT_DATA).filter(d =>
    fs.statSync(path.join(AMDGT_DATA, d)).isDirectory(),
  );
}

export function readJson(filePath: string): any | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}
