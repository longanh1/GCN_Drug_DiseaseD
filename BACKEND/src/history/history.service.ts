import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

const HISTORY_FILE = path.resolve(__dirname, '../../../AI_ENGINE/data/prediction_history.json');

@Injectable()
export class HistoryService {
  private history: any[] = [];

  constructor() {
    this._load();
  }

  private _load() {
    if (fs.existsSync(HISTORY_FILE)) {
      try {
        this.history = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
      } catch {
        this.history = [];
      }
    }
  }

  private _save() {
    const dir = path.dirname(HISTORY_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(this.history, null, 2));
  }

  addEntry(entry: any): any {
    const record = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...entry,
    };
    this.history.unshift(record);
    if (this.history.length > 200) this.history = this.history.slice(0, 200);
    this._save();
    return record;
  }

  getAll(limit = 50): any[] {
    return this.history.slice(0, limit);
  }

  clear() {
    this.history = [];
    this._save();
    return { message: 'History cleared' };
  }
}
