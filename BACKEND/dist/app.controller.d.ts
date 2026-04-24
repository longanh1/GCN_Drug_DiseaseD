import { AppService } from './app.service';
export declare class AppController {
    private readonly appService;
    constructor(appService: AppService);
    getRoot(): {
        message: string;
        version: string;
        status: string;
    };
    health(): {
        status: string;
        timestamp: string;
    };
}
