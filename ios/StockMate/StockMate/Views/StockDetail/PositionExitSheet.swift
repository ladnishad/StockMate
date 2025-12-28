import SwiftUI

/// Sheet for logging position exits
struct PositionExitSheet: View {
    @ObservedObject var viewModel: StockDetailViewModel
    @Environment(\.dismiss) private var dismiss

    // Exit type
    @State private var exitType: ExitType = .partial

    // Exit fields
    @State private var exitPrice: String = ""
    @State private var sharesToExit: String = ""
    @State private var exitReason: ExitReason = .manual

    // Re-evaluation confirmation
    @State private var showReEvalConfirmation = false

    enum ExitType: String, CaseIterable {
        case partial = "Partial"
        case full = "Full"
    }

    enum ExitReason: String, CaseIterable {
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case stopLoss = "stop_loss"
        case manual = "manual"

        var displayName: String {
            switch self {
            case .target1: return "Target 1 Hit"
            case .target2: return "Target 2 Hit"
            case .target3: return "Target 3 Hit"
            case .stopLoss: return "Stopped Out"
            case .manual: return "Manual Exit"
            }
        }
    }

    private var position: Position? {
        viewModel.position
    }

    private var currentPrice: Double? {
        viewModel.detail?.currentPrice ?? position?.currentPrice
    }

    private var maxShares: Int {
        position?.currentSize ?? 0
    }

    var body: some View {
        NavigationView {
            Form {
                // Exit type picker
                Section {
                    Picker("Exit Type", selection: $exitType) {
                        ForEach(ExitType.allCases, id: \.self) { type in
                            Text(type.rawValue).tag(type)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: exitType) { _, newValue in
                        if newValue == .full {
                            sharesToExit = "\(maxShares)"
                        }
                    }
                }

                // Exit details
                Section("Exit Details") {
                    HStack {
                        Text("Exit Price")
                        Spacer()
                        TextField("0.00", text: $exitPrice)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                    }

                    if let price = currentPrice {
                        HStack {
                            Text("Current Price")
                                .foregroundColor(.secondary)
                            Spacer()
                            Button {
                                exitPrice = String(format: "%.2f", price)
                            } label: {
                                Text(String(format: "$%.2f", price))
                                    .foregroundColor(.blue)
                            }
                        }
                    }

                    if exitType == .partial {
                        HStack {
                            Text("Shares to Exit")
                            Spacer()
                            TextField("0", text: $sharesToExit)
                                .keyboardType(.numberPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }

                        HStack {
                            Text("Position Size")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(maxShares) shares")
                                .foregroundColor(.secondary)
                        }
                    }

                    // Reason picker
                    Picker("Reason", selection: $exitReason) {
                        ForEach(ExitReason.allCases, id: \.self) { reason in
                            Text(reason.displayName).tag(reason)
                        }
                    }
                }

                // P&L Preview
                if let preview = pnlPreview {
                    Section("P&L Preview") {
                        HStack {
                            Text("Entry")
                            Spacer()
                            Text("\(preview.entryFormatted) x \(preview.shares)")
                                .foregroundColor(.secondary)
                        }

                        HStack {
                            Text("Exit")
                            Spacer()
                            Text("\(preview.exitFormatted) x \(preview.shares)")
                                .foregroundColor(.secondary)
                        }

                        Divider()

                        HStack {
                            Text("P&L")
                                .fontWeight(.semibold)
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text(preview.pnlFormatted)
                                    .fontWeight(.bold)
                                    .foregroundColor(preview.isProfit ? .green : .red)
                                Text(preview.pnlPctFormatted)
                                    .font(.caption)
                                    .foregroundColor(preview.isProfit ? .green : .red)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Close Position")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Confirm") {
                        Task {
                            await submitExit()
                            // Only show re-eval confirmation if exit succeeded
                            if viewModel.positionError == nil {
                                showReEvalConfirmation = true
                            } else {
                                dismiss()
                            }
                        }
                    }
                    .disabled(!isValid)
                }
            }
            .onAppear {
                prefillWithCurrentPrice()
            }
            .alert("Re-evaluate Trading Plan?", isPresented: $showReEvalConfirmation) {
                Button("Yes") {
                    NotificationCenter.default.post(
                        name: .planEvaluationTriggered,
                        object: nil,
                        userInfo: ["symbol": viewModel.symbol]
                    )
                    dismiss()
                }
                Button("No", role: .cancel) {
                    dismiss()
                }
            } message: {
                Text("Update the AI analysis based on your position?")
            }
        }
    }

    // MARK: - P&L Preview

    private struct PnLPreview {
        let entryPrice: Double
        let exitPrice: Double
        let shares: Int
        let pnl: Double
        let pnlPct: Double

        var entryFormatted: String {
            String(format: "$%.2f", entryPrice)
        }

        var exitFormatted: String {
            String(format: "$%.2f", exitPrice)
        }

        var pnlFormatted: String {
            let sign = pnl >= 0 ? "+" : ""
            return String(format: "%@$%.2f", sign, pnl)
        }

        var pnlPctFormatted: String {
            let sign = pnlPct >= 0 ? "+" : ""
            return String(format: "%@%.2f%%", sign, pnlPct)
        }

        var isProfit: Bool {
            pnl >= 0
        }
    }

    private var pnlPreview: PnLPreview? {
        guard let position = position,
              let avgEntry = position.avgEntryPrice,
              let exit = Double(exitPrice),
              let shares = exitType == .full ? maxShares : Int(sharesToExit),
              shares > 0 else {
            return nil
        }

        let costBasis = avgEntry * Double(shares)
        let proceeds = exit * Double(shares)
        let pnl = proceeds - costBasis
        let pnlPct = (pnl / costBasis) * 100

        return PnLPreview(
            entryPrice: avgEntry,
            exitPrice: exit,
            shares: shares,
            pnl: pnl,
            pnlPct: pnlPct
        )
    }

    // MARK: - Validation

    private var isValid: Bool {
        guard let _ = Double(exitPrice) else { return false }

        let shares = exitType == .full ? maxShares : (Int(sharesToExit) ?? 0)
        return shares > 0 && shares <= maxShares
    }

    // MARK: - Actions

    private func prefillWithCurrentPrice() {
        if let price = currentPrice {
            exitPrice = String(format: "%.2f", price)
        }

        // Default to full exit
        sharesToExit = "\(maxShares)"
    }

    private func submitExit() async {
        guard let price = Double(exitPrice) else { return }

        let shares = exitType == .full ? maxShares : (Int(sharesToExit) ?? 0)
        guard shares > 0 else { return }

        await viewModel.addExit(
            price: price,
            shares: shares,
            reason: exitReason.rawValue
        )
    }
}
