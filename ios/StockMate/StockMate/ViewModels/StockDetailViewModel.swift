import SwiftUI
import Combine

/// ViewModel for the stock detail page
@MainActor
class StockDetailViewModel: ObservableObject {
    // MARK: - Published State

    @Published var detail: StockDetail?
    @Published var selectedTimeframe: ChartTimeframe = .oneDay
    @Published var isLoading = false
    @Published var isLoadingBars = false
    @Published var isLoadingAnalysis = true  // Separate loading state for analysis section
    @Published var error: String?
    @Published var isInWatchlist = false
    @Published var isTogglingWatchlist = false

    // Position tracking
    @Published var position: Position?
    @Published var isLoadingPosition = false
    @Published var positionError: String?
    @Published var showPositionEntrySheet = false
    @Published var showPositionExitSheet = false

    // MARK: - Properties

    let symbol: String
    private let watchlistService = WatchlistService.shared
    private let apiService = APIService.shared

    /// Cache for bars loaded on-demand
    private var barsCache: [ChartTimeframe: [PriceBar]] = [:]

    // MARK: - Computed Properties

    /// Chart bars for the selected timeframe
    var chartBars: [PriceBar] {
        // Check cache first
        if let cached = barsCache[selectedTimeframe] {
            return cached
        }

        // Fallback to pre-loaded data from detail (only 1M and 1W are pre-loaded)
        guard let detail = detail else { return [] }
        switch selectedTimeframe {
        case .oneMonth:
            return detail.bars1d
        case .oneWeek:
            return detail.bars1h
        default:
            // All other timeframes (including 1D) need to be fetched on-demand
            return []
        }
    }

    /// Whether there's chart data available
    var hasChartData: Bool {
        !chartBars.isEmpty
    }

    /// Price range for the current chart
    var priceRange: ClosedRange<Double>? {
        guard !chartBars.isEmpty else { return nil }
        let lows = chartBars.map { $0.low }
        let highs = chartBars.map { $0.high }
        guard let minLow = lows.min(), let maxHigh = highs.max() else { return nil }
        // Add some padding
        let padding = (maxHigh - minLow) * 0.1
        return (minLow - padding)...(maxHigh + padding)
    }

    // MARK: - Initialization

    init(symbol: String) {
        self.symbol = symbol
        self.isInWatchlist = watchlistService.hasSymbol(symbol)
    }

    // MARK: - Data Loading

    /// Load all data - starts chart bars loading immediately in parallel with detail
    func loadAllData() async {
        guard detail == nil else { return } // Don't reload if already loaded

        isLoading = true
        isLoadingAnalysis = true
        isLoadingBars = true
        error = nil

        // Start loading bars immediately (don't wait for detail)
        // This makes the chart appear faster
        async let barsTask: () = loadBarsInBackground()

        // Load detail (contains analysis - slower)
        async let detailTask: () = loadDetailData()

        // Wait for both to complete
        await barsTask
        await detailTask

        isLoading = false
    }

    /// Load stock detail from API
    private func loadDetailData() async {
        do {
            let fetchedDetail = try await watchlistService.getStockDetail(symbol: symbol)
            withAnimation {
                self.detail = fetchedDetail
                self.isLoadingAnalysis = false
            }
            // Update watchlist status
            isInWatchlist = watchlistService.hasSymbol(symbol)
        } catch {
            self.error = error.localizedDescription
            self.isLoadingAnalysis = false
            // Use sample data for preview/development
            #if DEBUG
            self.detail = StockDetail.sample
            #endif
        }
    }

    /// Load bars in background - called in parallel with detail loading
    private func loadBarsInBackground() async {
        // Load bars for the default timeframe immediately
        do {
            let bars = try await apiService.fetchBars(symbol: symbol, timeframe: selectedTimeframe)
            barsCache[selectedTimeframe] = bars
            print("📊 Pre-loaded \(bars.count) bars for \(selectedTimeframe.rawValue)")
        } catch {
            print("📊 Failed to pre-load bars: \(error)")
        }
        isLoadingBars = false
    }

    /// Legacy method - kept for compatibility
    func loadDetail() async {
        await loadAllData()
    }

    /// Refresh stock detail
    func refresh() async {
        detail = nil
        barsCache.removeAll()
        await loadAllData()
    }

    // MARK: - Watchlist Actions

    /// Toggle watchlist membership
    func toggleWatchlist() async {
        isTogglingWatchlist = true

        do {
            if isInWatchlist {
                try await watchlistService.removeSymbol(symbol)
                withAnimation {
                    isInWatchlist = false
                }
            } else {
                _ = try await watchlistService.addSymbol(symbol)
                withAnimation {
                    isInWatchlist = true
                }
            }

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            // Haptic feedback for error
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)

            self.error = error.localizedDescription
        }

        isTogglingWatchlist = false
    }

    // MARK: - Timeframe Selection

    /// Called when timeframe changes - loads bars if needed
    func onTimeframeChanged(_ timeframe: ChartTimeframe) {
        // Haptic feedback
        let generator = UISelectionFeedbackGenerator()
        generator.selectionChanged()

        // Load bars on-demand if not in cache
        Task {
            await loadBarsIfNeeded(for: timeframe)
        }
    }

    /// Load bars for a timeframe if not already cached
    private func loadBarsIfNeeded(for timeframe: ChartTimeframe) async {
        // Check if already cached
        if barsCache[timeframe] != nil {
            return
        }

        // Check if it's a pre-loaded timeframe from detail (only 1M and 1W)
        switch timeframe {
        case .oneMonth, .oneWeek:
            // These are pre-loaded in detail, no need to fetch
            return
        default:
            break
        }

        // Load on-demand
        isLoadingBars = true
        print("📊 Loading bars for \(timeframe.rawValue)...")

        do {
            let bars = try await apiService.fetchBars(symbol: symbol, timeframe: timeframe)
            print("📊 Received \(bars.count) bars for \(timeframe.rawValue)")
            barsCache[timeframe] = bars
        } catch {
            print("📊 Failed to load bars for \(timeframe.rawValue): \(error)")
            // Don't show error to user, just keep showing empty chart
        }

        isLoadingBars = false
    }

    // MARK: - Position Management

    /// Whether the user has an active position
    var hasPosition: Bool {
        guard let position = position else { return false }
        return position.status != .closed && position.status != .stoppedOut
    }

    /// Whether the user has entered the trade (has shares)
    var hasEnteredPosition: Bool {
        position?.hasPosition ?? false
    }

    /// Load position with live P&L
    func loadPosition() async {
        isLoadingPosition = true
        positionError = nil

        do {
            position = try await apiService.getPositionWithPnl(symbol: symbol)
        } catch {
            print("Failed to load position: \(error)")
            // Don't show error - position might not exist
        }

        isLoadingPosition = false
    }

    /// Create a new position (starts in "watching" status)
    func createPosition(
        tradeType: String = "swing",
        stopLoss: Double? = nil,
        target1: Double? = nil,
        target2: Double? = nil,
        target3: Double? = nil
    ) async {
        isLoadingPosition = true
        positionError = nil

        do {
            position = try await apiService.createPosition(
                symbol: symbol,
                tradeType: tradeType,
                stopLoss: stopLoss,
                target1: target1,
                target2: target2,
                target3: target3
            )

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            positionError = error.localizedDescription
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)
        }

        isLoadingPosition = false
    }

    /// Add an entry to the position (scale in or initial entry)
    func addEntry(price: Double, shares: Int) async {
        isLoadingPosition = true
        positionError = nil

        do {
            position = try await apiService.addPositionEntry(
                symbol: symbol,
                price: price,
                shares: shares
            )

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            positionError = error.localizedDescription
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)
        }

        isLoadingPosition = false
    }

    /// Add an exit from the position (partial or full)
    func addExit(price: Double, shares: Int, reason: String = "manual") async {
        isLoadingPosition = true
        positionError = nil

        do {
            position = try await apiService.addPositionExit(
                symbol: symbol,
                price: price,
                shares: shares,
                reason: reason
            )

            // Haptic feedback
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            positionError = error.localizedDescription
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)
        }

        isLoadingPosition = false
    }

    /// Close the entire position
    func closePosition(exitPrice: Double, reason: String = "manual") async {
        guard let currentSize = position?.currentSize, currentSize > 0 else { return }
        await addExit(price: exitPrice, shares: currentSize, reason: reason)
    }

    /// Delete the position tracking
    func deletePosition() async {
        isLoadingPosition = true

        do {
            try await apiService.deletePosition(symbol: symbol)
            position = nil

            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
        } catch {
            positionError = error.localizedDescription
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.error)
        }

        isLoadingPosition = false
    }

    /// Refresh position with latest P&L
    func refreshPosition() async {
        guard position != nil else { return }
        await loadPosition()
    }
}
