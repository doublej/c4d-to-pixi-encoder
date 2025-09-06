//#region src/debug.d.ts
/**
 * Represents a pricing discrepancy between original and calculated costs
 */
type Discrepancy = {
  file: string;
  timestamp: string;
  model: string;
  originalCost: number;
  calculatedCost: number;
  difference: number;
  percentDiff: number;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
};
/**
 * Statistics about pricing mismatches across all usage data
 */
type MismatchStats = {
  totalEntries: number;
  entriesWithBoth: number;
  matches: number;
  mismatches: number;
  discrepancies: Discrepancy[];
  modelStats: Map<string, {
    total: number;
    matches: number;
    mismatches: number;
    avgPercentDiff: number;
  }>;
  versionStats: Map<string, {
    total: number;
    matches: number;
    mismatches: number;
    avgPercentDiff: number;
  }>;
};
/**
 * Analyzes usage data to detect pricing mismatches between stored and calculated costs
 * Compares pre-calculated costUSD values with costs calculated from token usage
 * @param claudePath - Optional path to Claude data directory
 * @returns Statistics about pricing mismatches found
 */
declare function detectMismatches(claudePath?: string): Promise<MismatchStats>;
/**
 * Prints a detailed report of pricing mismatches to the console
 * @param stats - Mismatch statistics to report
 * @param sampleCount - Number of sample discrepancies to show (default: 5)
 */
declare function printMismatchReport(stats: MismatchStats, sampleCount?: number): void;
//#endregion
export { detectMismatches, printMismatchReport };