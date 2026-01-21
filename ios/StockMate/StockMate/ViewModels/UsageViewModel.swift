import Foundation
import SwiftUI
import Combine

/// ViewModel for admin usage dashboard
@MainActor
class UsageViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var isAdmin: Bool = false
    @Published var isLoading: Bool = false
    @Published var isCheckingAdmin: Bool = true
    @Published var error: String?

    // Summary data
    @Published var summary: UsageSummary?
    @Published var userSummaries: [UserUsageSummary] = []
    @Published var dailyCosts: [DailyCostItem] = []
    @Published var operationBreakdowns: [OperationTypeBreakdown] = []

    // Filters
    @Published var selectedDays: Int = 30
    @Published var selectedUserId: String?

    // Time period options
    let dayOptions = [7, 14, 30, 60, 90]

    // MARK: - Computed Properties

    /// Total cost formatted as currency
    var formattedTotalCost: String {
        guard let summary = summary else { return "$0.00" }
        return formatCurrency(summary.totalCost)
    }

    /// Claude cost percentage
    var claudePercentage: Double {
        guard let summary = summary, summary.totalCost > 0 else { return 0 }
        return (summary.claudeCost / summary.totalCost) * 100
    }

    /// Grok cost percentage
    var grokPercentage: Double {
        guard let summary = summary, summary.totalCost > 0 else { return 0 }
        return (summary.grokCost / summary.totalCost) * 100
    }

    /// Average daily cost
    var averageDailyCost: Double {
        guard !dailyCosts.isEmpty else { return 0 }
        return dailyCosts.reduce(0) { $0 + $1.cost } / Double(dailyCosts.count)
    }

    /// Peak daily cost
    var peakDailyCost: Double {
        dailyCosts.map { $0.cost }.max() ?? 0
    }

    /// Most expensive operation
    var topOperation: OperationTypeBreakdown? {
        operationBreakdowns.max(by: { $0.totalCost < $1.totalCost })
    }

    /// Top user by cost
    var topUser: UserUsageSummary? {
        userSummaries.first
    }

    // MARK: - API Methods

    /// Check if current user is admin
    func checkAdminStatus() async {
        isCheckingAdmin = true

        do {
            let response = try await APIService.shared.checkAdminStatus()
            isAdmin = response.isAdmin
        } catch {
            // Not admin or error - either way, don't show admin UI
            isAdmin = false
            print("UsageViewModel: Not admin or error - \(error)")
        }

        isCheckingAdmin = false
    }

    /// Load all usage data
    func loadAllData() async {
        guard isAdmin else { return }

        isLoading = true
        error = nil

        // Load all data in parallel
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await self.loadSummary() }
            group.addTask { await self.loadUserSummaries() }
            group.addTask { await self.loadDailyCosts() }
            group.addTask { await self.loadOperationBreakdowns() }
        }

        isLoading = false
    }

    /// Load summary data
    func loadSummary() async {
        do {
            let response = try await APIService.shared.getUsageSummary(
                userId: selectedUserId,
                days: selectedDays
            )
            summary = response.summary
        } catch {
            self.error = "Failed to load summary"
            print("UsageViewModel: Error loading summary - \(error)")
        }
    }

    /// Load user summaries
    func loadUserSummaries() async {
        do {
            let response = try await APIService.shared.getUsageByUser(
                days: selectedDays,
                limit: 50
            )
            userSummaries = response.users
        } catch {
            print("UsageViewModel: Error loading user summaries - \(error)")
        }
    }

    /// Load daily costs
    func loadDailyCosts() async {
        do {
            let response = try await APIService.shared.getDailyCosts(
                userId: selectedUserId,
                days: selectedDays
            )
            dailyCosts = response.dailyCosts.reversed() // Chronological order
        } catch {
            print("UsageViewModel: Error loading daily costs - \(error)")
        }
    }

    /// Load operation breakdowns
    func loadOperationBreakdowns() async {
        do {
            let response = try await APIService.shared.getUsageByOperation(
                userId: selectedUserId,
                days: selectedDays
            )
            operationBreakdowns = response.breakdowns
        } catch {
            print("UsageViewModel: Error loading operation breakdowns - \(error)")
        }
    }

    /// Refresh data when filter changes
    func refreshData() async {
        await loadAllData()
    }

    // MARK: - Formatting Helpers

    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = value < 1 ? 4 : 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }

    func formatNumber(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? "0"
    }

    func formatCompactNumber(_ value: Int) -> String {
        if value >= 1_000_000 {
            return String(format: "%.1fM", Double(value) / 1_000_000)
        } else if value >= 1_000 {
            return String(format: "%.1fK", Double(value) / 1_000)
        }
        return "\(value)"
    }

    func formatPercentage(_ value: Double) -> String {
        String(format: "%.1f%%", value)
    }
}
