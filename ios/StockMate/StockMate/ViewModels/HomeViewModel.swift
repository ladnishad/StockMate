import SwiftUI
import Combine

/// ViewModel for the Home screen managing market data and watchlist
@MainActor
class HomeViewModel: ObservableObject {
    // MARK: - Published State

    @Published var indices: [MarketIndex] = []
    @Published var selectedProfile: TraderProfile = .swingTrader {
        didSet {
            if oldValue != selectedProfile {
                Task { await loadWatchlist() }
            }
        }
    }
    @Published var watchlistStocks: [Stock] = []
    @Published var marketDirection: MarketDirection = .mixed

    @Published var isLoadingIndices = false
    @Published var isLoadingWatchlist = false
    @Published var isRefreshing = false
    @Published var indicesError: String?
    @Published var watchlistError: String?
    @Published var error: String?

    @Published var lastUpdated: Date?

    // MARK: - Computed Properties

    var isLoading: Bool {
        isLoadingIndices || isLoadingWatchlist
    }

    var hasError: Bool {
        indicesError != nil || watchlistError != nil || error != nil
    }

    var formattedLastUpdated: String {
        guard let lastUpdated else { return "Never" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: lastUpdated, relativeTo: Date())
    }

    // MARK: - Initialization

    init() {
        // Load persisted profile preference
        if let savedProfile = UserDefaults.standard.string(forKey: "selectedProfile"),
           let profile = TraderProfile(rawValue: savedProfile) {
            selectedProfile = profile
        }
    }

    // MARK: - Data Loading

    /// Load initial data on app launch
    func loadInitialData() async {
        await loadAllData()
    }

    /// Load all data (indices + watchlist)
    func loadAllData() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.loadMarketIndices() }
            group.addTask { await self.loadWatchlist() }
        }
        lastUpdated = Date()
    }

    /// Refresh all data
    func refresh() async {
        isRefreshing = true
        error = nil
        await loadAllData()
        isRefreshing = false
    }

    /// Dismiss current error
    func dismissError() {
        error = nil
        indicesError = nil
        watchlistError = nil
    }

    /// Load market indices
    func loadMarketIndices() async {
        isLoadingIndices = true
        indicesError = nil

        do {
            let fetchedIndices = try await APIService.shared.fetchMarketQuick()
            withAnimation(.easeInOut(duration: 0.3)) {
                self.indices = fetchedIndices
                self.updateMarketDirection()
            }
        } catch {
            indicesError = error.localizedDescription
            // Use sample data for development/preview
            if indices.isEmpty {
                indices = MarketIndex.samples
                updateMarketDirection()
            }
        }

        isLoadingIndices = false
    }

    /// Load watchlist based on current profile
    func loadWatchlist() async {
        isLoadingWatchlist = true
        watchlistError = nil

        do {
            let stocks = try await APIService.shared.fetchWatchlist(profile: selectedProfile)
            withAnimation(.easeInOut(duration: 0.3)) {
                self.watchlistStocks = stocks
            }
        } catch {
            watchlistError = error.localizedDescription
            // Use sample data for development/preview
            if watchlistStocks.isEmpty {
                watchlistStocks = Stock.samples
            }
        }

        isLoadingWatchlist = false
    }

    // MARK: - Profile Management

    /// Change the selected profile
    func changeProfile(_ profile: TraderProfile) {
        guard profile != selectedProfile else { return }

        // Haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        // Save preference
        UserDefaults.standard.set(profile.rawValue, forKey: "selectedProfile")

        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
            selectedProfile = profile
        }
    }

    // MARK: - Private Helpers

    private func updateMarketDirection() {
        let upCount = indices.filter { $0.isUp }.count
        let total = indices.count

        if total == 0 {
            marketDirection = .mixed
        } else if upCount >= (total * 3 / 4) {
            marketDirection = .bullish
        } else if upCount <= (total / 4) {
            marketDirection = .bearish
        } else {
            marketDirection = .mixed
        }
    }
}

// MARK: - Supporting Types

enum MarketDirection: String {
    case bullish = "Bullish"
    case bearish = "Bearish"
    case mixed = "Mixed"

    var icon: String {
        switch self {
        case .bullish: return "arrow.up.right"
        case .bearish: return "arrow.down.right"
        case .mixed: return "arrow.left.arrow.right"
        }
    }

    var color: Color {
        switch self {
        case .bullish: return .green
        case .bearish: return .red
        case .mixed: return .orange
        }
    }
}
