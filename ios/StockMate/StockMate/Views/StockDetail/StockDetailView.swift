import SwiftUI

/// Redesigned stock detail view - Clean Apple-like aesthetic for novice traders
/// The app acts as an expert trader guiding the user through the analysis
struct StockDetailView: View {
    @StateObject private var viewModel: StockDetailViewModel
    @StateObject private var planViewModel: TradingPlanViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme
    @State private var showingChat = false
    @State private var showingPlan = false

    init(symbol: String) {
        _viewModel = StateObject(wrappedValue: StockDetailViewModel(symbol: symbol))
        _planViewModel = StateObject(wrappedValue: TradingPlanViewModel(symbol: symbol))
    }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
        ScrollView(.vertical, showsIndicators: false) {
            if let error = viewModel.error, viewModel.detail == nil {
                // Only show error if we have no data at all
                ErrorStateView(message: error) {
                    Task { await viewModel.refresh() }
                }
            } else {
                VStack(spacing: 0) {
                    // Hero Section: Price + Chart (shows skeleton or data)
                    if let detail = viewModel.detail {
                        PriceHeroSection(
                            detail: detail,
                            selectedTimeframe: $viewModel.selectedTimeframe,
                            chartBars: viewModel.chartBars,
                            isLoadingBars: viewModel.isLoadingBars
                        )

                        // Key Statistics (compact, right after chart)
                        KeyStatsRow(detail: detail)
                            .padding(.horizontal, 20)
                            .padding(.bottom, 16)
                    } else {
                        // Chart skeleton while loading
                        PriceHeroSkeleton(
                            symbol: viewModel.symbol,
                            chartBars: viewModel.chartBars,
                            isLoadingBars: viewModel.isLoadingBars,
                            selectedTimeframe: $viewModel.selectedTimeframe
                        )

                        // Stats skeleton
                        KeyStatsRowSkeleton()
                            .padding(.horizontal, 20)
                            .padding(.bottom, 16)
                    }

                    // Main Content
                    VStack(spacing: 16) {
                        // Trading Levels - minimal display when plan exists
                        TradingLevelsCard(planViewModel: planViewModel)

                        // Position Card
                        if let position = viewModel.position {
                            PositionCard(position: position, viewModel: viewModel)
                        } else if viewModel.isInWatchlist && !viewModel.isLoadingPosition {
                            // Show "Track Position" button for watchlist stocks
                            TrackPositionButton(viewModel: viewModel, plan: planViewModel.plan)
                        }

                        // Key Levels (Support/Resistance)
                        if let detail = viewModel.detail,
                           !detail.supportLevels.isEmpty || !detail.resistanceLevels.isEmpty {
                            KeyLevelsSection(
                                supportLevels: detail.supportLevels,
                                resistanceLevels: detail.resistanceLevels,
                                currentPrice: detail.currentPrice
                            )
                        }

                        // Note: Key Insights section removed - AI thesis in ExpertGuidanceCard now provides insights

                        // Disclaimer
                        DisclaimerFooter()
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 40)
                }
            }
        }

        // Floating AI Chat Button
        AIChatButton {
            showingChat = true
        }
        .padding(.trailing, 20)
        .padding(.bottom, 24)
        }
        .background(Color(.systemBackground))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                VStack(spacing: 0) {
                    Text(viewModel.symbol)
                        .font(.system(size: 17, weight: .semibold))
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                PlanButton(hasPlan: planViewModel.plan != nil) {
                    showingPlan = true
                }
            }
        }
        .refreshable {
            await viewModel.refresh()
        }
        .task {
            await viewModel.loadDetail()
        }
        .fullScreenCover(isPresented: $showingChat) {
            NavigationStack {
                ChatView(symbol: viewModel.symbol)
            }
        }
        .sheet(isPresented: $showingPlan) {
            NavigationStack {
                SimplifiedPlanView(symbol: viewModel.symbol)
                    .navigationTitle("Trading Plan")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") {
                                showingPlan = false
                            }
                            .fontWeight(.medium)
                        }
                    }
            }
        }
        .task {
            await planViewModel.loadPlan()
        }
        .task {
            await viewModel.loadPosition()
        }
        .onChange(of: viewModel.selectedTimeframe) { _, newTimeframe in
            viewModel.onTimeframeChanged(newTimeframe)
        }
        .sheet(isPresented: $viewModel.showPositionEntrySheet) {
            PositionEntrySheet(viewModel: viewModel, hasPlan: planViewModel.plan != nil)
        }
        .sheet(isPresented: $viewModel.showPositionExitSheet) {
            PositionExitSheet(viewModel: viewModel)
        }
    }
}

// MARK: - AI Chat Button

private struct AIChatButton: View {
    let action: () -> Void
    @State private var isPressed = false
    @State private var pulseScale: CGFloat = 1.0

    var body: some View {
        Button(action: {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            action()
        }) {
            ZStack {
                // Outer glow pulse
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                Color.blue.opacity(0.3),
                                Color.blue.opacity(0.0)
                            ],
                            center: .center,
                            startRadius: 20,
                            endRadius: 40
                        )
                    )
                    .frame(width: 72, height: 72)
                    .scaleEffect(pulseScale)

                // Main button
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.0, green: 0.48, blue: 1.0),
                                Color(red: 0.0, green: 0.35, blue: 0.9)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 56, height: 56)
                    .shadow(color: Color.blue.opacity(0.4), radius: 12, x: 0, y: 6)

                // Icon
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundColor(.white)
            }
            .scaleEffect(isPressed ? 0.92 : 1.0)
        }
        .buttonStyle(.plain)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    withAnimation(.spring(response: 0.2, dampingFraction: 0.6)) {
                        isPressed = true
                    }
                }
                .onEnded { _ in
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                        isPressed = false
                    }
                }
        )
        .onAppear {
            withAnimation(
                .easeInOut(duration: 2.0)
                .repeatForever(autoreverses: true)
            ) {
                pulseScale = 1.15
            }
        }
    }
}

// MARK: - Price Hero Section

private struct PriceHeroSection: View {
    let detail: StockDetail
    @Binding var selectedTimeframe: ChartTimeframe
    let chartBars: [PriceBar]
    var isLoadingBars: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            // Company name and price
            VStack(spacing: 6) {
                Text(detail.name)
                    .font(.system(size: 15, weight: .regular))
                    .foregroundStyle(.secondary)

                Text(detail.formattedPrice)
                    .font(.system(size: 48, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText())

                // Change badge
                HStack(spacing: 4) {
                    Image(systemName: detail.isUp ? "arrow.up" : "arrow.down")
                        .font(.system(size: 12, weight: .bold))

                    Text("\(detail.formattedChange) (\(detail.formattedChangePct))")
                        .font(.system(size: 15, weight: .medium, design: .monospaced))
                }
                .foregroundStyle(detail.isUp ? Color(.systemGreen) : Color(.systemRed))
            }
            .padding(.top, 8)
            .padding(.bottom, 16)

            // Chart
            MinimalChart(
                bars: chartBars,
                isUp: detail.isUp,
                supportLevels: detail.supportLevels,
                resistanceLevels: detail.resistanceLevels,
                timeframe: selectedTimeframe
            )
            .frame(height: 200)
            .padding(.horizontal, 16)

            // Timeframe selector
            TimeframePills(selected: $selectedTimeframe, isLoading: isLoadingBars)
                .padding(.top, 12)
                .padding(.bottom, 16)
        }
        .background(Color(.systemBackground))
    }
}

// MARK: - Trading Levels Card (Minimal)

private struct TradingLevelsCard: View {
    @ObservedObject var planViewModel: TradingPlanViewModel

    private var isGenerating: Bool { planViewModel.isLoading || planViewModel.isStreaming }

    var body: some View {
        if isGenerating {
            // Minimal loading state
            HStack(spacing: 10) {
                ProgressView()
                    .scaleEffect(0.8)
                Text("Generating plan...")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.secondarySystemBackground))
            )
        } else if let plan = planViewModel.plan {
            // Show trading levels
            TradingLevelsDisplay(plan: plan)
        } else {
            // Subtle generate button
            GeneratePlanButton(planViewModel: planViewModel)
        }
    }
}

// MARK: - Trading Levels Display

private struct TradingLevelsDisplay: View {
    let plan: TradingPlanResponse

    private var isBearish: Bool { plan.bias.lowercased() == "bearish" }

    var body: some View {
        VStack(spacing: 0) {
            // Compact header
            HStack {
                Text("Trading Plan")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)

                Spacer()

                // Direction pill
                HStack(spacing: 4) {
                    Image(systemName: isBearish ? "arrow.down.right" : "arrow.up.right")
                        .font(.system(size: 9, weight: .bold))
                    Text(isBearish ? "SHORT" : "LONG")
                        .font(.system(size: 9, weight: .bold))
                }
                .foregroundStyle(isBearish ? Color(.systemRed) : Color(.systemGreen))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(isBearish ? Color(.systemRed).opacity(0.12) : Color(.systemGreen).opacity(0.12))
                )
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 12)

            // Levels grid
            VStack(spacing: 8) {
                // Entry Zone
                if let low = plan.entryZoneLow, let high = plan.entryZoneHigh {
                    LevelRow(
                        label: "Entry",
                        value: low == high ? formatPrice(low) : "\(formatPrice(low)) – \(formatPrice(high))",
                        color: Color(.systemBlue)
                    )
                }

                // Stop Loss
                if let stop = plan.stopLoss {
                    LevelRow(
                        label: "Stop",
                        value: formatPrice(stop),
                        color: Color(.systemRed)
                    )
                }

                // Targets
                if let t1 = plan.target1 {
                    LevelRow(
                        label: "Target 1",
                        value: formatPrice(t1),
                        color: Color(.systemGreen)
                    )
                }
                if let t2 = plan.target2 {
                    LevelRow(
                        label: "Target 2",
                        value: formatPrice(t2),
                        color: Color(.systemGreen).opacity(0.8)
                    )
                }
                if let t3 = plan.target3 {
                    LevelRow(
                        label: "Target 3",
                        value: formatPrice(t3),
                        color: Color(.systemGreen).opacity(0.6)
                    )
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 14)
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private func formatPrice(_ price: Double) -> String {
        String(format: "$%.2f", price)
    }
}

private struct LevelRow: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack {
            // Color indicator
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)

            Text(label)
                .font(.system(size: 14))
                .foregroundStyle(.secondary)

            Spacer()

            Text(value)
                .font(.system(size: 14, weight: .medium, design: .monospaced))
                .foregroundStyle(.primary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Generate Plan Button

private struct GeneratePlanButton: View {
    @ObservedObject var planViewModel: TradingPlanViewModel

    var body: some View {
        Button {
            Task {
                await planViewModel.generateNewPlanWithStreaming()
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 13))
                Text("Generate Trading Plan")
                    .font(.system(size: 14, weight: .medium))
            }
            .foregroundStyle(Color(.systemBlue))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.systemBlue).opacity(0.1))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Color(.systemBlue).opacity(0.2), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Key Statistics Row (Compact)

private struct KeyStatsRow: View {
    let detail: StockDetail

    var body: some View {
        VStack(spacing: 0) {
            // First row: Open, High, Low, Vol
            HStack(spacing: 0) {
                CompactStatItem(label: "Open", value: detail.formattedOpen)
                Divider().frame(height: 32)
                CompactStatItem(label: "High", value: detail.formattedHigh)
                Divider().frame(height: 32)
                CompactStatItem(label: "Low", value: detail.formattedLow)
                Divider().frame(height: 32)
                CompactStatItem(label: "Vol", value: detail.formattedVolume)
            }

            Divider()
                .padding(.horizontal, 12)

            // Second row: 52W High, 52W Low, Avg Vol
            HStack(spacing: 0) {
                CompactStatItem(label: "52W H", value: detail.formatted52WeekHigh)
                Divider().frame(height: 32)
                CompactStatItem(label: "52W L", value: detail.formatted52WeekLow)
                Divider().frame(height: 32)
                CompactStatItem(label: "Avg Vol", value: detail.formattedAvgVolume)
            }
        }
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

private struct CompactStatItem: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
    }
}

// MARK: - Key Levels Section

private struct KeyLevelsSection: View {
    let supportLevels: [Double]
    let resistanceLevels: [Double]
    let currentPrice: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Key Levels")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .tracking(0.5)
                .padding(.horizontal, 16)
                .padding(.top, 16)

            HStack(spacing: 12) {
                // Resistance
                VStack(alignment: .leading, spacing: 8) {
                    Label("Resistance", systemImage: "arrow.up.to.line")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)

                    ForEach(resistanceLevels.prefix(3), id: \.self) { level in
                        HStack {
                            Text(String(format: "$%.2f", level))
                                .font(.system(size: 14, weight: .medium, design: .monospaced))

                            Spacer()

                            let pctAway = ((level - currentPrice) / currentPrice) * 100
                            Text("+\(String(format: "%.1f", pctAway))%")
                                .font(.system(size: 12, weight: .regular, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }

                    if resistanceLevels.isEmpty {
                        Text("—")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(.tertiarySystemBackground))
                )

                // Support
                VStack(alignment: .leading, spacing: 8) {
                    Label("Support", systemImage: "arrow.down.to.line")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)

                    ForEach(supportLevels.prefix(3), id: \.self) { level in
                        HStack {
                            Text(String(format: "$%.2f", level))
                                .font(.system(size: 14, weight: .medium, design: .monospaced))

                            Spacer()

                            let pctAway = ((level - currentPrice) / currentPrice) * 100
                            Text("\(String(format: "%.1f", pctAway))%")
                                .font(.system(size: 12, weight: .regular, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }

                    if supportLevels.isEmpty {
                        Text("—")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(.tertiarySystemBackground))
                )
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

// MARK: - Minimal Chart

private struct MinimalChart: View {
    let bars: [PriceBar]
    let isUp: Bool
    let supportLevels: [Double]
    let resistanceLevels: [Double]
    var timeframe: ChartTimeframe = .oneMonth

    @State private var selectedIndex: Int?

    private var lineColor: Color {
        isUp ? Color(.systemGreen) : Color(.systemRed)
    }

    /// Format time for tooltip based on timeframe
    private func formatTime(for bar: PriceBar) -> String {
        guard let date = bar.date else { return "" }
        let formatter = DateFormatter()
        formatter.timeZone = TimeZone(identifier: "America/New_York")

        switch timeframe {
        case .oneDay:
            formatter.dateFormat = "h:mm a"
            return formatter.string(from: date)
        case .oneWeek:
            formatter.dateFormat = "E h:mm a"
            return formatter.string(from: date)
        case .oneMonth, .threeMonths:
            formatter.dateFormat = "MMM d"
            return formatter.string(from: date)
        case .sixMonths, .oneYear, .yearToDate:
            formatter.dateFormat = "MMM d, yyyy"
            return formatter.string(from: date)
        case .fiveYears, .all:
            formatter.dateFormat = "MMM yyyy"
            return formatter.string(from: date)
        }
    }

    /// Get session type for a bar (for 1D timeframe)
    private func sessionType(for bar: PriceBar) -> SessionType? {
        guard timeframe == .oneDay, let date = bar.date else { return nil }
        let calendar = Calendar.current
        let components = calendar.dateComponents(in: TimeZone(identifier: "America/New_York")!, from: date)
        let hour = components.hour ?? 0
        let minute = components.minute ?? 0
        let timeInMinutes = hour * 60 + minute

        // Pre-market: 4:00 AM - 9:30 AM ET (240 - 570 minutes)
        // Market hours: 9:30 AM - 4:00 PM ET (570 - 960 minutes)
        // After-hours: 4:00 PM - 8:00 PM ET (960 - 1200 minutes)

        if timeInMinutes < 570 {
            return .preMarket
        } else if timeInMinutes < 960 {
            return .marketHours
        } else {
            return .afterHours
        }
    }

    /// Calculate session boundaries for x-axis labels
    private func sessionBoundaries(width: CGFloat) -> [(x: CGFloat, label: String, startX: CGFloat, endX: CGFloat)] {
        guard timeframe == .oneDay, !bars.isEmpty else { return [] }

        var boundaries: [(x: CGFloat, label: String, startX: CGFloat, endX: CGFloat)] = []
        let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width

        var preMarketStart: Int?
        var preMarketEnd: Int?
        var marketStart: Int?
        var marketEnd: Int?
        var afterHoursStart: Int?
        var afterHoursEnd: Int?

        for (index, bar) in bars.enumerated() {
            if let session = sessionType(for: bar) {
                switch session {
                case .preMarket:
                    if preMarketStart == nil { preMarketStart = index }
                    preMarketEnd = index
                case .marketHours:
                    if marketStart == nil { marketStart = index }
                    marketEnd = index
                case .afterHours:
                    if afterHoursStart == nil { afterHoursStart = index }
                    afterHoursEnd = index
                }
            }
        }

        // Add pre-market boundary
        if let start = preMarketStart, let end = preMarketEnd {
            let startX = CGFloat(start) * stepX
            let endX = CGFloat(end) * stepX
            let centerX = (startX + endX) / 2
            boundaries.append((centerX, "Pre", startX, endX))
        }

        // Add market hours boundary
        if let start = marketStart, let end = marketEnd {
            let startX = CGFloat(start) * stepX
            let endX = CGFloat(end) * stepX
            let centerX = (startX + endX) / 2
            boundaries.append((centerX, "Market", startX, endX))
        }

        // Add after-hours boundary
        if let start = afterHoursStart, let end = afterHoursEnd {
            let startX = CGFloat(start) * stepX
            let endX = CGFloat(end) * stepX
            let centerX = (startX + endX) / 2
            boundaries.append((centerX, "After", startX, endX))
        }

        return boundaries
    }

    private enum SessionType {
        case preMarket, marketHours, afterHours
    }

    var body: some View {
        if bars.isEmpty {
            EmptyChartPlaceholder()
        } else {
            GeometryReader { geometry in
                let width = geometry.size.width
                let height = timeframe == .oneDay ? geometry.size.height - 24 : geometry.size.height  // Reserve space for labels

                let prices = bars.map { $0.close }
                let minPrice = (prices.min() ?? 0) * 0.995
                let maxPrice = (prices.max() ?? 100) * 1.005
                let priceRange = maxPrice - minPrice

                VStack(spacing: 0) {
                    ZStack {
                        // Session background shading for 1D
                        if timeframe == .oneDay {
                            let boundaries = sessionBoundaries(width: width)
                            ForEach(Array(boundaries.enumerated()), id: \.offset) { _, boundary in
                                Rectangle()
                                    .fill(boundary.label == "Market" ? Color(.systemGray6).opacity(0.5) : Color.clear)
                                    .frame(width: boundary.endX - boundary.startX)
                                    .position(x: (boundary.startX + boundary.endX) / 2, y: height / 2)
                            }
                        }

                        // Area fill
                        Path { path in
                            guard !bars.isEmpty else { return }

                            let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width

                            path.move(to: CGPoint(x: 0, y: height))

                            for (index, bar) in bars.enumerated() {
                                let x = CGFloat(index) * stepX
                                let y = height - ((CGFloat(bar.close) - CGFloat(minPrice)) / CGFloat(priceRange) * height)

                                if index == 0 {
                                    path.addLine(to: CGPoint(x: x, y: y))
                                } else {
                                    path.addLine(to: CGPoint(x: x, y: y))
                                }
                            }

                            path.addLine(to: CGPoint(x: width, y: height))
                            path.closeSubpath()
                        }
                        .fill(
                            LinearGradient(
                                colors: [lineColor.opacity(0.3), lineColor.opacity(0.0)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )

                        // Line
                        Path { path in
                            guard !bars.isEmpty else { return }

                            let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width

                            for (index, bar) in bars.enumerated() {
                                let x = CGFloat(index) * stepX
                                let y = height - ((CGFloat(bar.close) - CGFloat(minPrice)) / CGFloat(priceRange) * height)

                                if index == 0 {
                                    path.move(to: CGPoint(x: x, y: y))
                                } else {
                                    path.addLine(to: CGPoint(x: x, y: y))
                                }
                            }
                        }
                        .stroke(lineColor, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))

                        // Current price dot
                        if let lastBar = bars.last {
                            let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width
                            let x = CGFloat(bars.count - 1) * stepX
                            let y = height - ((CGFloat(lastBar.close) - CGFloat(minPrice)) / CGFloat(priceRange) * height)

                            Circle()
                                .fill(lineColor)
                                .frame(width: 8, height: 8)
                                .position(x: x, y: y)
                        }

                        // Selected point indicator
                        if let index = selectedIndex, index < bars.count {
                            let bar = bars[index]
                            let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width
                            let x = CGFloat(index) * stepX
                            let y = height - ((CGFloat(bar.close) - CGFloat(minPrice)) / CGFloat(priceRange) * height)

                            // Vertical line
                            Rectangle()
                                .fill(Color(.systemGray3))
                                .frame(width: 1, height: height)
                                .position(x: x, y: height / 2)

                            // Dot
                            Circle()
                                .fill(Color(.label))
                                .frame(width: 10, height: 10)
                                .position(x: x, y: y)

                            // Price and time label
                            VStack(spacing: 2) {
                                Text(String(format: "$%.2f", bar.close))
                                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                Text(formatTime(for: bar))
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color(.systemGray6))
                            .cornerRadius(4)
                            .position(x: min(max(x, 50), width - 50), y: max(y - 30, 24))
                        }
                    }
                    .frame(height: height)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                let stepX = bars.count > 1 ? width / CGFloat(bars.count - 1) : width
                                let index = Int((value.location.x / stepX).rounded())
                                selectedIndex = max(0, min(index, bars.count - 1))
                            }
                            .onEnded { _ in
                                withAnimation(.easeOut(duration: 0.2)) {
                                    selectedIndex = nil
                                }
                            }
                    )

                    // Session labels for 1D timeframe
                    if timeframe == .oneDay {
                        HStack(spacing: 0) {
                            let boundaries = sessionBoundaries(width: width)
                            ForEach(Array(boundaries.enumerated()), id: \.offset) { idx, boundary in
                                Text(boundary.label)
                                    .font(.system(size: 9, weight: .medium))
                                    .foregroundStyle(boundary.label == "Market" ? .primary : .secondary)
                                    .frame(width: boundary.endX - boundary.startX)
                                if idx < boundaries.count - 1 {
                                    Rectangle()
                                        .fill(Color(.systemGray4))
                                        .frame(width: 1, height: 12)
                                }
                            }
                        }
                        .frame(height: 20)
                        .padding(.top, 4)
                    }
                }
            }
        }
    }
}

private struct EmptyChartPlaceholder: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.tertiary)

            Text("No data available")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Timeframe Pills

private struct TimeframePills: View {
    @Binding var selected: ChartTimeframe
    var isLoading: Bool = false

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(ChartTimeframe.allCases) { timeframe in
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selected = timeframe
                        }
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    } label: {
                        Text(timeframe.rawValue)
                            .font(.system(size: 13, weight: selected == timeframe ? .semibold : .medium))
                            .foregroundStyle(selected == timeframe ? .primary : .secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                Capsule()
                                    .fill(selected == timeframe ? Color(.systemGray5) : Color.clear)
                            )
                            .overlay {
                                if isLoading && selected == timeframe {
                                    ProgressView()
                                        .scaleEffect(0.6)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 4)
        }
    }
}

// MARK: - Disclaimer Footer

private struct DisclaimerFooter: View {
    var body: some View {
        Text("This analysis is for educational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.")
            .font(.system(size: 11, weight: .regular))
            .foregroundStyle(.tertiary)
            .multilineTextAlignment(.center)
            .padding(.top, 8)
    }
}

// MARK: - Loading State

private struct LoadingStateView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)

            Text("Analyzing...")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 400)
    }
}

// MARK: - Skeleton Components

/// Skeleton for price hero section - shows chart if bars available, otherwise loading
private struct PriceHeroSkeleton: View {
    let symbol: String
    let chartBars: [PriceBar]
    let isLoadingBars: Bool
    @Binding var selectedTimeframe: ChartTimeframe

    var body: some View {
        VStack(spacing: 0) {
            // Company name and price skeleton
            VStack(spacing: 6) {
                SkeletonView(width: 100, height: 15)

                SkeletonView(width: 140, height: 48)

                SkeletonView(width: 80, height: 20)
            }
            .padding(.top, 8)
            .padding(.bottom, 16)

            // Show actual chart if we have bars, otherwise skeleton
            if !chartBars.isEmpty {
                MinimalChart(
                    bars: chartBars,
                    isUp: true,
                    supportLevels: [],
                    resistanceLevels: [],
                    timeframe: selectedTimeframe
                )
                .frame(height: 200)
                .padding(.horizontal, 16)
            } else if isLoadingBars {
                // Chart loading state
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.systemGray6))
                        .frame(height: 200)

                    VStack(spacing: 8) {
                        ProgressView()
                        Text("Loading chart...")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 16)
            } else {
                // Empty chart placeholder
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray6))
                    .frame(height: 200)
                    .padding(.horizontal, 16)
            }

            // Timeframe selector
            TimeframePills(selected: $selectedTimeframe, isLoading: isLoadingBars)
                .padding(.top, 12)
                .padding(.bottom, 16)
        }
        .background(Color(.systemBackground))
    }
}

/// Skeleton for key stats row
private struct KeyStatsRowSkeleton: View {
    var body: some View {
        HStack(spacing: 12) {
            ForEach(0..<5, id: \.self) { _ in
                VStack(spacing: 4) {
                    SkeletonView(width: 30, height: 12)
                    SkeletonView(width: 50, height: 16)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }
}

/// Reusable skeleton view with shimmer effect
private struct SkeletonView: View {
    let width: CGFloat
    let height: CGFloat
    @State private var isAnimating = false

    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color(.systemGray5))
            .frame(width: width, height: height)
            .opacity(isAnimating ? 0.5 : 1.0)
            .animation(
                .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                value: isAnimating
            )
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - Plan Button

private struct PlanButton: View {
    let hasPlan: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            action()
        }) {
            Image(systemName: "list.clipboard")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(hasPlan ? .primary : .secondary)
        }
    }
}

// MARK: - Preview

#Preview("Stock Detail - Buy") {
    NavigationStack {
        StockDetailView(symbol: "AAPL")
    }
}

#Preview("Stock Detail - Hold") {
    NavigationStack {
        StockDetailView(symbol: "TSLA")
    }
}
