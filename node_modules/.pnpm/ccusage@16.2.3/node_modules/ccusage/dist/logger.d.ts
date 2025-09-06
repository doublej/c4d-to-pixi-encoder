import { ConsolaInstance } from "consola";

//#region src/logger.d.ts

/**
 * Application logger instance with package name tag
 */
declare const logger: ConsolaInstance;
/**
 * Direct console.log function for cases where logger formatting is not desired
 */
// eslint-disable-next-line no-console
declare const log: {
  (...data: any[]): void;
  (message?: any, ...optionalParams: any[]): void;
};
//#endregion
export { log, logger };