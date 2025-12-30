import SwiftUI

/// Card displaying the user's position and P&L
struct PositionCard: View {
    let position: Position
    @ObservedObject var viewModel: StockDetailViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Label("YOUR POSITION", systemImage: "chart.line.uptrend.xyaxis")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)

                Spacer()

                // Status badge
                Text(position.status.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(statusColor.opacity(0.15))
                    .foregroundColor(statusColor)
                    .cornerRadius(4)

                // Menu
                Menu {
                    if position.hasPosition {
                        Button {
                            viewModel.showPositionEntrySheet = true
                        } label: {
                            Label("Scale In", systemImage: "plus.circle")
                        }

                        Button {
                            viewModel.showPositionExitSheet = true
                        } label: {
                            Label("Scale Out", systemImage: "minus.circle")
                        }

                        Divider()
                    }

                    Button(role: .destructive) {
                        Task {
                            await viewModel.deletePosition()
                        }
                    } label: {
                        Label("Delete Position", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .foregroundColor(.secondary)
                }
            }

            if position.hasPosition {
                // Entry info
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Entry")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(position.avgEntryFormatted)
                            .font(.title3)
                            .fontWeight(.semibold)
                        Text("\(position.currentSize) shares")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    if let currentPrice = position.currentPrice {
                        VStack(alignment: .trailing, spacing: 4) {
                            Text("Current")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "$%.2f", currentPrice))
                                .font(.title3)
                                .fontWeight(.semibold)
                            if let pct = position.unrealizedPnlPct {
                                Text(String(format: "%@%.2f%%", pct >= 0 ? "+" : "", pct))
                                    .font(.caption)
                                    .foregroundColor(pct >= 0 ? .green : .red)
                            }
                        }
                    }
                }

                // P&L boxes
                HStack(spacing: 12) {
                    PnlBox(
                        title: "Unrealized",
                        value: position.unrealizedPnlFormatted,
                        percentage: position.unrealizedPnlPctFormatted,
                        isPositive: (position.unrealizedPnl ?? 0) >= 0
                    )

                    PnlBox(
                        title: "Realized",
                        value: position.realizedPnlFormatted,
                        percentage: position.realizedPnlPctFormatted,
                        isPositive: (position.realizedPnl ?? 0) >= 0
                    )
                }

                // Levels
                HStack(spacing: 16) {
                    // Stop loss
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Stop")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        HStack(spacing: 4) {
                            Text(position.stopLossFormatted)
                                .font(.caption)
                                .fontWeight(.medium)
                            if let distance = position.stopLossDistance {
                                Text("(\(String(format: "%.1f%%", distance)))")
                                    .font(.caption2)
                                    .foregroundColor(.red)
                            }
                        }
                    }

                    // Targets
                    ForEach(position.targets, id: \.number) { target in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 2) {
                                Text("T\(target.number)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                if target.hit {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.caption2)
                                        .foregroundColor(.green)
                                }
                            }
                            Text(String(format: "$%.2f", target.price))
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(target.hit ? .green : .primary)
                        }
                    }

                    Spacer()
                }

                // Action buttons
                HStack(spacing: 12) {
                    Button {
                        viewModel.showPositionEntrySheet = true
                    } label: {
                        Label("Scale In", systemImage: "plus")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        viewModel.showPositionExitSheet = true
                    } label: {
                        Label("Scale Out", systemImage: "minus")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    .buttonStyle(.bordered)

                    Spacer()
                }
            } else {
                // Watching mode - show "Enter Trade" button
                VStack(spacing: 12) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Stop Loss")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(position.stopLossFormatted)
                                .font(.headline)
                        }

                        Spacer()

                        ForEach(position.targets, id: \.number) { target in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Target \(target.number)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "$%.2f", target.price))
                                    .font(.headline)
                            }
                        }
                    }

                    Button {
                        viewModel.showPositionEntrySheet = true
                    } label: {
                        Label("Log Entry", systemImage: "plus.circle.fill")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            // Entries history (if multiple)
            if position.entries.count > 1 {
                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    Text("Entry History")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)

                    ForEach(position.entries) { entry in
                        HStack {
                            Text(entry.formattedDate)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(entry.shares) @ \(entry.formattedPrice)")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                    }
                }
            }

            // Exits history (if any)
            if !position.exits.isEmpty {
                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    Text("Exit History")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)

                    ForEach(position.exits) { exit in
                        HStack {
                            Text(exit.reasonDisplay)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(exit.shares) @ \(exit.formattedPrice)")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.05), radius: 8, y: 2)
    }

    private var statusColor: Color {
        switch position.status {
        case .watching: return .gray
        case .entered: return .blue
        case .partial: return .orange
        case .stoppedOut: return .red
        case .closed: return .green
        }
    }
}

/// P&L display box
private struct PnlBox: View {
    let title: String
    let value: String
    let percentage: String
    let isPositive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(isPositive ? .green : .red)
            if !percentage.isEmpty {
                Text(percentage)
                    .font(.caption)
                    .foregroundColor(isPositive ? .green : .red)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(isPositive ? Color.green.opacity(0.1) : Color.red.opacity(0.1))
        .cornerRadius(8)
    }
}

/// Button to start tracking a position (shown when no position exists)
struct TrackPositionButton: View {
    @ObservedObject var viewModel: StockDetailViewModel
    let plan: TradingPlanResponse?

    var body: some View {
        Button {
            if let plan = plan {
                // Auto-create position from plan
                Task {
                    await viewModel.createPosition(
                        tradeType: plan.tradeStyle ?? "swing",
                        stopLoss: plan.stopLoss,
                        target1: plan.target1,
                        target2: plan.target2,
                        target3: plan.target3
                    )
                }
            } else {
                // Show entry sheet for manual entry
                viewModel.showPositionEntrySheet = true
            }
        } label: {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                Text("Track Position")
                    .fontWeight(.medium)
            }
            .font(.subheadline)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .buttonStyle(.bordered)
        .disabled(viewModel.isLoadingPosition)
    }
}
