import Foundation
import SwiftUI
import Combine

/// View model for the Scanner view
@MainActor
class ScannerViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var selectedStyle: TradingStyle = .day
    @Published var dayResults: [ScannerResult] = []
    @Published var swingResults: [ScannerResult] = []
    @Published var positionResults: [ScannerResult] = []

    @Published var isLoading: Bool = false
    @Published var isRefreshing: Bool = false
    @Published var error: String?

    @Published var lastScanTime: Date?
    @Published var nextScheduledScan: Date?

    private let scannerService = ScannerService.shared
    private let watchlistService = WatchlistService.shared

    // MARK: - Computed Properties

    var currentResults: [ScannerResult] {
        switch selectedStyle {
        case .day: return dayResults
        case .swing: return swingResults
        case .position: return positionResults
        }
    }

    var isEmpty: Bool {
        currentResults.isEmpty
    }

    var formattedLastScanTime: String {
        guard let time = lastScanTime else { return "Never" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: time, relativeTo: Date())
    }

    // MARK: - Data Loading

    /// Load scanner results from API
    func loadResults() async {
        guard !isLoading else { return }

        isLoading = true
        error = nil

        do {
            let response = try await scannerService.getAllResults()

            // Update results
            dayResults = updateWatchingStatus(response.day.results)
            swingResults = updateWatchingStatus(response.swing.results)
            positionResults = updateWatchingStatus(response.position.results)

            lastScanTime = response.scanTime
            nextScheduledScan = response.nextScheduledScan

        } catch {
            self.error = error.localizedDescription
            print("Scanner load error: \(error)")
        }

        isLoading = false
    }

    /// Refresh scanner (trigger on-demand scan)
    func refresh() async {
        guard !isRefreshing else { return }

        isRefreshing = true
        error = nil

        do {
            let response = try await scannerService.refresh()

            // Update results
            dayResults = updateWatchingStatus(response.day.results)
            swingResults = updateWatchingStatus(response.swing.results)
            positionResults = updateWatchingStatus(response.position.results)

            lastScanTime = response.scanTime
            nextScheduledScan = response.nextScheduledScan

        } catch {
            self.error = error.localizedDescription
            print("Scanner refresh error: \(error)")
        }

        isRefreshing = false
    }

    /// Update isWatching status based on user's watchlist
    private func updateWatchingStatus(_ results: [ScannerResult]) -> [ScannerResult] {
        return results.map { result in
            var updated = result
            // Check if symbol is in watchlist using the service
            let isInWatchlist = watchlistService.hasSymbol(result.symbol)

            // Create a new result with updated isWatching flag
            return ScannerResult(
                symbol: result.symbol,
                style: result.style,
                confidenceGrade: result.confidenceGrade,
                confidenceScore: result.confidenceScore,
                currentPrice: result.currentPrice,
                description: result.description,
                patternType: result.patternType,
                keyLevels: result.keyLevels,
                detectedAt: result.detectedAt,
                isNew: result.isNew,
                isWatching: isInWatchlist,
                warnings: result.warnings,
                volumeMultiple: result.volumeMultiple,
                gapPct: result.gapPct,
                fibLevel: result.fibLevel,
                rsiValue: result.rsiValue,
                vwap: result.vwap,
                expiresAt: result.expiresAt
            )
        }
    }

    // MARK: - Actions

    /// Add a scanned stock to watchlist
    func addToWatchlist(_ result: ScannerResult) async -> Bool {
        let scannerSource = "\(result.style.displayName) Trade Scanner"
        let scannerReason = result.patternType.displayName

        do {
            _ = try await scannerService.addToWatchlist(
                symbol: result.symbol,
                scannerSource: scannerSource,
                scannerReason: scannerReason
            )

            // Update local state to reflect the change
            updateResultWatchingStatus(symbol: result.symbol, isWatching: true)

            return true
        } catch {
            self.error = "Failed to add to watchlist: \(error.localizedDescription)"
            return false
        }
    }

    /// Update the watching status for a specific symbol across all results
    private func updateResultWatchingStatus(symbol: String, isWatching: Bool) {
        dayResults = dayResults.map { result in
            guard result.symbol == symbol else { return result }
            return ScannerResult(
                symbol: result.symbol,
                style: result.style,
                confidenceGrade: result.confidenceGrade,
                confidenceScore: result.confidenceScore,
                currentPrice: result.currentPrice,
                description: result.description,
                patternType: result.patternType,
                keyLevels: result.keyLevels,
                detectedAt: result.detectedAt,
                isNew: result.isNew,
                isWatching: isWatching,
                warnings: result.warnings,
                volumeMultiple: result.volumeMultiple,
                gapPct: result.gapPct,
                fibLevel: result.fibLevel,
                rsiValue: result.rsiValue,
                vwap: result.vwap,
                expiresAt: result.expiresAt
            )
        }

        swingResults = swingResults.map { result in
            guard result.symbol == symbol else { return result }
            return ScannerResult(
                symbol: result.symbol,
                style: result.style,
                confidenceGrade: result.confidenceGrade,
                confidenceScore: result.confidenceScore,
                currentPrice: result.currentPrice,
                description: result.description,
                patternType: result.patternType,
                keyLevels: result.keyLevels,
                detectedAt: result.detectedAt,
                isNew: result.isNew,
                isWatching: isWatching,
                warnings: result.warnings,
                volumeMultiple: result.volumeMultiple,
                gapPct: result.gapPct,
                fibLevel: result.fibLevel,
                rsiValue: result.rsiValue,
                vwap: result.vwap,
                expiresAt: result.expiresAt
            )
        }

        positionResults = positionResults.map { result in
            guard result.symbol == symbol else { return result }
            return ScannerResult(
                symbol: result.symbol,
                style: result.style,
                confidenceGrade: result.confidenceGrade,
                confidenceScore: result.confidenceScore,
                currentPrice: result.currentPrice,
                description: result.description,
                patternType: result.patternType,
                keyLevels: result.keyLevels,
                detectedAt: result.detectedAt,
                isNew: result.isNew,
                isWatching: isWatching,
                warnings: result.warnings,
                volumeMultiple: result.volumeMultiple,
                gapPct: result.gapPct,
                fibLevel: result.fibLevel,
                rsiValue: result.rsiValue,
                vwap: result.vwap,
                expiresAt: result.expiresAt
            )
        }
    }

    /// Dismiss error
    func dismissError() {
        error = nil
    }
}
