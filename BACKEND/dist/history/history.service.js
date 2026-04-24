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
exports.HistoryService = void 0;
const common_1 = require("@nestjs/common");
const fs = require("fs");
const path = require("path");
const HISTORY_FILE = path.resolve(__dirname, '../../../AI_ENGINE/data/prediction_history.json');
let HistoryService = class HistoryService {
    constructor() {
        this.history = [];
        this._load();
    }
    _load() {
        if (fs.existsSync(HISTORY_FILE)) {
            try {
                this.history = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
            }
            catch {
                this.history = [];
            }
        }
    }
    _save() {
        const dir = path.dirname(HISTORY_FILE);
        if (!fs.existsSync(dir))
            fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(HISTORY_FILE, JSON.stringify(this.history, null, 2));
    }
    addEntry(entry) {
        const record = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            ...entry,
        };
        this.history.unshift(record);
        if (this.history.length > 200)
            this.history = this.history.slice(0, 200);
        this._save();
        return record;
    }
    getAll(limit = 50) {
        return this.history.slice(0, limit);
    }
    clear() {
        this.history = [];
        this._save();
        return { message: 'History cleared' };
    }
};
exports.HistoryService = HistoryService;
exports.HistoryService = HistoryService = __decorate([
    (0, common_1.Injectable)(),
    __metadata("design:paramtypes", [])
], HistoryService);
//# sourceMappingURL=history.service.js.map