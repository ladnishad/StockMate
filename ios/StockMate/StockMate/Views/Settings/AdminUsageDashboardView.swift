import SwiftUI
import Charts

/// Admin dashboard for viewing API usage and costs
struct AdminUsageDashboardView: View {
    @StateObject private var viewModel = UsageViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Time Period Filter
                    periodFilterSection

                    if viewModel.isLoading {
                        loadingView
                    } else if let error = viewModel.error {
                        errorView(error)
                    } else {
                        // Summary Cards
                        summaryCardsSection

                        // Cost Trend Chart
                        costTrendSection

                        // Provider Breakdown
                        providerBreakdownSection

                        // Operation Breakdown
                        operationBreakdownSection

                        // Top Users
                        topUsersSection
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 40)
            }
            .scrollIndicators(.hidden)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Usage Dashboard")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }

                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        Task { await viewModel.refreshData() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(viewModel.isLoading)
                }
            }
            .task {
                await viewModel.checkAdminStatus()
                if viewModel.isAdmin {
                    await viewModel.loadAllData()
                }
            }
        }
    }

    // MARK: - Period Filter

    private var periodFilterSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "calendar")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text("Time Period")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            .padding(.leading, 4)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(viewModel.dayOptions, id: \.self) { days in
                        PeriodChip(
                            days: days,
                            isSelected: viewModel.selectedDays == days
                        ) {
                            viewModel.selectedDays = days
                            Task { await viewModel.refreshData() }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Summary Cards

    private var summaryCardsSection: some View {
        VStack(spacing: 12) {
            // Main cost card
            TotalCostCard(viewModel: viewModel)

            // Stats row
            HStack(spacing: 12) {
                StatCard(
                    title: "Requests",
                    value: viewModel.formatCompactNumber(viewModel.summary?.totalRequests ?? 0),
                    icon: "arrow.up.arrow.down",
                    color: .blue
                )

                StatCard(
                    title: "Tokens",
                    value: viewModel.formatCompactNumber(viewModel.summary?.totalTokens ?? 0),
                    icon: "character.cursor.ibeam",
                    color: .purple
                )

                StatCard(
                    title: "Avg/Day",
                    value: viewModel.formatCurrency(viewModel.averageDailyCost),
                    icon: "chart.line.uptrend.xyaxis",
                    color: .orange
                )
            }
        }
    }

    // MARK: - Cost Trend Chart

    private var costTrendSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionHeader(title: "Cost Trend", icon: "chart.xyaxis.line")

            VStack(alignment: .leading, spacing: 12) {
                if viewModel.dailyCosts.isEmpty {
                    emptyChartPlaceholder
                } else {
                    CostTrendChart(data: viewModel.dailyCosts)
                        .frame(height: 200)
                }

                // Legend
                HStack(spacing: 20) {
                    ChartLegendItem(color: .blue, label: "Claude")
                    ChartLegendItem(color: .white, label: "Grok")
                }
                .padding(.horizontal, 8)
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - Provider Breakdown

    private var providerBreakdownSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionHeader(title: "Provider Breakdown", icon: "cpu")

            VStack(spacing: 16) {
                if let summary = viewModel.summary, summary.totalCost > 0 {
                    ProviderDonutChart(
                        claudeCost: summary.claudeCost,
                        grokCost: summary.grokCost
                    )
                    .frame(height: 180)

                    // Provider details
                    VStack(spacing: 12) {
                        ProviderDetailRow(
                            name: "Claude",
                            company: "Anthropic",
                            cost: summary.claudeCost,
                            requests: summary.claudeRequests,
                            tokens: summary.claudeInputTokens + summary.claudeOutputTokens,
                            color: claudeColor,
                            viewModel: viewModel
                        )

                        Divider()

                        ProviderDetailRow(
                            name: "Grok",
                            company: "xAI",
                            cost: summary.grokCost,
                            requests: summary.grokRequests,
                            tokens: summary.grokInputTokens + summary.grokOutputTokens,
                            color: .white,
                            viewModel: viewModel
                        )
                    }
                } else {
                    emptyChartPlaceholder
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - Operation Breakdown

    private var operationBreakdownSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionHeader(title: "By Operation Type", icon: "square.stack.3d.up")

            VStack(spacing: 0) {
                if viewModel.operationBreakdowns.isEmpty {
                    emptyChartPlaceholder
                        .padding(16)
                } else {
                    ForEach(Array(viewModel.operationBreakdowns.enumerated()), id: \.element.id) { index, breakdown in
                        OperationRow(breakdown: breakdown, viewModel: viewModel)

                        if index < viewModel.operationBreakdowns.count - 1 {
                            Divider()
                                .padding(.leading, 52)
                        }
                    }
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - Top Users

    private var topUsersSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionHeader(title: "Top Users by Cost", icon: "person.2")

            VStack(spacing: 0) {
                if viewModel.userSummaries.isEmpty {
                    emptyChartPlaceholder
                        .padding(16)
                } else {
                    ForEach(Array(viewModel.userSummaries.prefix(10).enumerated()), id: \.element.id) { index, user in
                        UserCostRow(user: user, rank: index + 1, viewModel: viewModel)

                        if index < min(viewModel.userSummaries.count, 10) - 1 {
                            Divider()
                                .padding(.leading, 52)
                        }
                    }
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
    }

    // MARK: - Helper Views

    private func sectionHeader(title: String, icon: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .tracking(0.5)
        }
        .padding(.leading, 4)
    }

    private var emptyChartPlaceholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)

            Text("No data available")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 120)
    }

    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)

            Text("Loading usage data...")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 300)
    }

    private func errorView(_ message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundStyle(.orange)

            Text(message)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button("Retry") {
                Task { await viewModel.loadAllData() }
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity)
        .padding(40)
    }

    // Claude brand color
    private var claudeColor: Color {
        Color(red: 0.851, green: 0.467, blue: 0.341)
    }
}

// MARK: - Period Chip

struct PeriodChip: View {
    let days: Int
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text("\(days)d")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(isSelected ? .white : .primary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    Capsule()
                        .fill(isSelected ? Color.blue : Color(.secondarySystemGroupedBackground))
                )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Total Cost Card

struct TotalCostCard: View {
    @ObservedObject var viewModel: UsageViewModel

    var body: some View {
        VStack(spacing: 8) {
            Text("Total Cost")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)

            Text(viewModel.formattedTotalCost)
                .font(.system(size: 42, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text("Last \(viewModel.selectedDays) days")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color.blue.opacity(0.15),
                            Color.purple.opacity(0.08)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.blue.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Stat Card

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(color)

            Text(value)
                .font(.system(size: 18, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)

            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

// MARK: - Cost Trend Chart

struct CostTrendChart: View {
    let data: [DailyCostItem]

    var body: some View {
        Chart {
            ForEach(data) { item in
                // Stacked area for Claude
                AreaMark(
                    x: .value("Date", item.parsedDate ?? Date()),
                    y: .value("Claude", item.claudeCost)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [Color.blue.opacity(0.6), Color.blue.opacity(0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .interpolationMethod(.catmullRom)

                // Line for total
                LineMark(
                    x: .value("Date", item.parsedDate ?? Date()),
                    y: .value("Total", item.cost)
                )
                .foregroundStyle(Color.primary.opacity(0.8))
                .lineStyle(StrokeStyle(lineWidth: 2))
                .interpolationMethod(.catmullRom)
            }
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: max(1, data.count / 5))) { value in
                AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                    .font(.system(size: 10))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading) { value in
                AxisValueLabel {
                    if let cost = value.as(Double.self) {
                        Text("$\(String(format: "%.2f", cost))")
                            .font(.system(size: 10, design: .monospaced))
                    }
                }
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [4, 4]))
            }
        }
    }
}

// MARK: - Provider Donut Chart

struct ProviderDonutChart: View {
    let claudeCost: Double
    let grokCost: Double

    private var claudeColor: Color {
        Color(red: 0.851, green: 0.467, blue: 0.341)
    }

    var body: some View {
        Chart {
            SectorMark(
                angle: .value("Cost", claudeCost),
                innerRadius: .ratio(0.6),
                angularInset: 2
            )
            .foregroundStyle(claudeColor)
            .cornerRadius(4)

            SectorMark(
                angle: .value("Cost", grokCost),
                innerRadius: .ratio(0.6),
                angularInset: 2
            )
            .foregroundStyle(Color.gray)
            .cornerRadius(4)
        }
        .chartBackground { proxy in
            GeometryReader { geo in
                let frame = geo[proxy.plotFrame!]
                VStack(spacing: 4) {
                    Text("Total")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text("$\(String(format: "%.2f", claudeCost + grokCost))")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                }
                .position(x: frame.midX, y: frame.midY)
            }
        }
    }
}

// MARK: - Chart Legend Item

struct ChartLegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color == .white ? Color.gray : color)
                .frame(width: 8, height: 8)

            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Provider Detail Row

struct ProviderDetailRow: View {
    let name: String
    let company: String
    let cost: Double
    let requests: Int
    let tokens: Int
    let color: Color
    @ObservedObject var viewModel: UsageViewModel

    var body: some View {
        HStack(spacing: 14) {
            // Color indicator
            RoundedRectangle(cornerRadius: 4)
                .fill(color == .white ? Color.gray : color)
                .frame(width: 4, height: 40)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(.system(size: 16, weight: .semibold))

                    Text(company)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.tertiary)
                }

                Text("\(viewModel.formatCompactNumber(requests)) requests \u{2022} \(viewModel.formatCompactNumber(tokens)) tokens")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(viewModel.formatCurrency(cost))
                .font(.system(size: 17, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
        }
    }
}

// MARK: - Operation Row

struct OperationRow: View {
    let breakdown: OperationTypeBreakdown
    @ObservedObject var viewModel: UsageViewModel

    private var operationColor: Color {
        switch breakdown.operationType {
        case "plan_generation": return .blue
        case "plan_evaluation": return .green
        case "chat": return .purple
        case "orchestrator": return .orange
        case "subagent": return .cyan
        case "image_analysis": return .pink
        default: return .gray
        }
    }

    var body: some View {
        HStack(spacing: 14) {
            // Icon
            ZStack {
                Circle()
                    .fill(operationColor.opacity(0.15))
                    .frame(width: 36, height: 36)

                Image(systemName: breakdown.icon)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(operationColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(breakdown.displayName)
                    .font(.system(size: 15, weight: .semibold))

                Text("\(viewModel.formatNumber(breakdown.requestCount)) requests")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(viewModel.formatCurrency(breakdown.totalCost))
                    .font(.system(size: 15, weight: .bold, design: .rounded))

                Text("\(viewModel.formatCurrency(breakdown.avgCostPerRequest))/req")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

// MARK: - User Cost Row

struct UserCostRow: View {
    let user: UserUsageSummary
    let rank: Int
    @ObservedObject var viewModel: UsageViewModel

    var body: some View {
        HStack(spacing: 14) {
            // Rank badge
            ZStack {
                Circle()
                    .fill(rankColor.opacity(0.15))
                    .frame(width: 36, height: 36)

                Text("\(rank)")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(rankColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(user.displayName)
                    .font(.system(size: 15, weight: .semibold))
                    .lineLimit(1)

                Text("\(viewModel.formatNumber(user.totalRequests)) requests")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(viewModel.formatCurrency(user.totalCost))
                    .font(.system(size: 15, weight: .bold, design: .rounded))

                // Provider split mini indicator
                HStack(spacing: 4) {
                    if user.claudeCost > 0 {
                        Circle()
                            .fill(Color(red: 0.851, green: 0.467, blue: 0.341))
                            .frame(width: 6, height: 6)
                    }
                    if user.grokCost > 0 {
                        Circle()
                            .fill(Color.gray)
                            .frame(width: 6, height: 6)
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var rankColor: Color {
        switch rank {
        case 1: return .yellow
        case 2: return .gray
        case 3: return .orange
        default: return .blue
        }
    }
}

// MARK: - Preview

#Preview {
    AdminUsageDashboardView()
}
