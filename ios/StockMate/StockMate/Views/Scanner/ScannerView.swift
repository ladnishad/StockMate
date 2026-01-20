import SwiftUI

/// Main scanner view with trading style tabs and results
struct ScannerView: View {
    @StateObject private var viewModel = ScannerViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Style picker
                ScannerStylePicker(selectedStyle: $viewModel.selectedStyle)
                    .padding(.horizontal, 20)
                    .padding(.top, 8)

                // Status bar
                ScanStatusBar(
                    lastScanTime: viewModel.formattedLastScanTime,
                    isRefreshing: viewModel.isRefreshing
                )
                .padding(.horizontal, 20)
                .padding(.top, 12)

                // Results list
                if viewModel.isLoading && viewModel.currentResults.isEmpty {
                    ScannerLoadingView()
                } else if viewModel.isEmpty {
                    ScannerEmptyState(onRefresh: {
                        Task { await viewModel.refresh() }
                    })
                } else {
                    ScannerResultsList(
                        results: viewModel.currentResults,
                        isRefreshing: viewModel.isRefreshing,
                        onRefresh: {
                            await viewModel.refresh()
                        },
                        onAdd: { result in
                            Task {
                                let success = await viewModel.addToWatchlist(result)
                                if success {
                                    // Haptic feedback
                                    let generator = UINotificationFeedbackGenerator()
                                    generator.notificationOccurred(.success)
                                }
                            }
                        }
                    )
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Scanner")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                    .fontWeight(.medium)
                }

                ToolbarItem(placement: .topBarTrailing) {
                    ScannerRefreshButton(
                        isRefreshing: viewModel.isRefreshing,
                        action: {
                            Task { await viewModel.refresh() }
                        }
                    )
                }
            }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("Retry") {
                    Task { await viewModel.loadResults() }
                }
                Button("Dismiss", role: .cancel) {
                    viewModel.dismissError()
                }
            } message: {
                if let error = viewModel.error {
                    Text(error)
                }
            }
        }
        .task {
            await viewModel.loadResults()
        }
    }
}

// MARK: - Supporting Components

/// Segmented style picker for scanner
struct ScannerStylePicker: View {
    @Binding var selectedStyle: TradingStyle

    var body: some View {
        HStack(spacing: 0) {
            ForEach(TradingStyle.allCases, id: \.self) { style in
                ScannerStyleTab(
                    title: style.displayName,
                    isSelected: selectedStyle == style,
                    action: {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            selectedStyle = style
                        }
                    }
                )
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Individual style tab for scanner
struct ScannerStyleTab: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(isSelected ? .white : .secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(
                    Group {
                        if isSelected {
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color.blue)
                        }
                    }
                )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

/// Status bar showing last scan time
struct ScanStatusBar: View {
    let lastScanTime: String
    let isRefreshing: Bool

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "clock")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)

            Text("Last scan: \(lastScanTime)")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer()

            if isRefreshing {
                HStack(spacing: 4) {
                    ProgressView()
                        .scaleEffect(0.7)

                    Text("Scanning...")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.blue)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Results list with pull-to-refresh
struct ScannerResultsList: View {
    let results: [ScannerResult]
    let isRefreshing: Bool
    let onRefresh: () async -> Void
    let onAdd: (ScannerResult) -> Void

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                ForEach(results) { result in
                    NavigationLink(value: result.symbol) {
                        ScannerResultCard(
                            result: result,
                            onAdd: { onAdd(result) },
                            onTap: { }
                        )
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 20)
        }
        .scrollIndicators(.hidden)
        .refreshable {
            await onRefresh()
        }
        .navigationDestination(for: String.self) { symbol in
            StockDetailView(symbol: symbol)
        }
    }
}

/// Loading view with skeletons for scanner
struct ScannerLoadingView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                ForEach(0..<5, id: \.self) { _ in
                    ScannerResultCardSkeleton()
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
        }
        .scrollIndicators(.hidden)
    }
}

/// Empty state when no scanner results
struct ScannerEmptyState: View {
    let onRefresh: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            // Icon
            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.system(size: 60))
                .foregroundStyle(.secondary.opacity(0.5))

            // Text
            VStack(spacing: 8) {
                Text("No Setups Found")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text("Markets may be consolidating.\nTry scanning again later.")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            // Refresh button
            Button(action: onRefresh) {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 14, weight: .semibold))

                    Text("Scan Now")
                        .font(.system(size: 15, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .background(
                    Capsule()
                        .fill(Color.blue)
                )
            }
            .buttonStyle(PlainButtonStyle())

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Refresh button for scanner toolbar
struct ScannerRefreshButton: View {
    let isRefreshing: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            if isRefreshing {
                ProgressView()
                    .scaleEffect(0.8)
            } else {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 15, weight: .semibold))
            }
        }
        .disabled(isRefreshing)
    }
}

// MARK: - Preview

#Preview("Scanner View") {
    ScannerView()
}

#Preview("Style Picker") {
    struct PreviewWrapper: View {
        @State var style: TradingStyle = .day

        var body: some View {
            ScannerStylePicker(selectedStyle: $style)
                .padding()
                .background(Color(.systemGroupedBackground))
        }
    }

    return PreviewWrapper()
}

#Preview("Empty State") {
    ScannerEmptyState(onRefresh: { })
        .background(Color(.systemGroupedBackground))
}
