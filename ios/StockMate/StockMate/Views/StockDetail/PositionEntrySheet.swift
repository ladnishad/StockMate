import SwiftUI

/// Sheet for logging position entries
struct PositionEntrySheet: View {
    @ObservedObject var viewModel: StockDetailViewModel
    @Environment(\.dismiss) private var dismiss

    // Entry mode
    @State private var entryMode: EntryMode = .quick

    // Quick entry fields
    @State private var avgPrice: String = ""
    @State private var totalShares: String = ""

    // Multiple entries fields
    @State private var entries: [EntryField] = [EntryField()]

    // Position setup (if creating new)
    @State private var stopLoss: String = ""
    @State private var target1: String = ""
    @State private var target2: String = ""
    @State private var target3: String = ""

    // Re-evaluation confirmation
    @State private var showReEvalConfirmation = false

    enum EntryMode: String, CaseIterable {
        case quick = "Quick Entry"
        case multiple = "Multiple Entries"
    }

    struct EntryField: Identifiable {
        let id = UUID()
        var price: String = ""
        var shares: String = ""
    }

    private var isNewPosition: Bool {
        viewModel.position == nil
    }

    private var currentPrice: Double? {
        viewModel.detail?.currentPrice ?? viewModel.position?.currentPrice
    }

    var body: some View {
        NavigationView {
            Form {
                // Entry mode picker
                Section {
                    Picker("Entry Mode", selection: $entryMode) {
                        ForEach(EntryMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                if entryMode == .quick {
                    quickEntrySection
                } else {
                    multipleEntriesSection
                }

                // Position setup (only for new positions)
                if isNewPosition {
                    positionSetupSection
                }

                // Summary
                summarySection
            }
            .navigationTitle(isNewPosition ? "Create Position" : "Add Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button(isNewPosition ? "Create" : "Add") {
                        Task {
                            await submitEntry()
                            // Only show re-eval confirmation if entry succeeded
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
                prefillFromPlan()
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

    // MARK: - Quick Entry Section

    private var quickEntrySection: some View {
        Section("Entry Details") {
            HStack {
                Text("Average Price")
                Spacer()
                TextField("0.00", text: $avgPrice)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }

            HStack {
                Text("Total Shares")
                Spacer()
                TextField("0", text: $totalShares)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }

            if let price = currentPrice {
                HStack {
                    Text("Current Price")
                        .foregroundColor(.secondary)
                    Spacer()
                    Button {
                        avgPrice = String(format: "%.2f", price)
                    } label: {
                        Text(String(format: "$%.2f", price))
                            .foregroundColor(.blue)
                    }
                }
            }
        }
    }

    // MARK: - Multiple Entries Section

    private var multipleEntriesSection: some View {
        Section("Entries") {
            ForEach($entries) { $entry in
                HStack {
                    VStack(alignment: .leading) {
                        Text("Price")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        TextField("0.00", text: $entry.price)
                            .keyboardType(.decimalPad)
                    }

                    VStack(alignment: .leading) {
                        Text("Shares")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        TextField("0", text: $entry.shares)
                            .keyboardType(.numberPad)
                    }

                    if entries.count > 1 {
                        Button {
                            entries.removeAll { $0.id == entry.id }
                        } label: {
                            Image(systemName: "minus.circle.fill")
                                .foregroundColor(.red)
                        }
                    }
                }
            }

            Button {
                entries.append(EntryField())
            } label: {
                Label("Add Another Entry", systemImage: "plus.circle")
            }
        }
    }

    // MARK: - Position Setup Section

    private var positionSetupSection: some View {
        Section("Risk Management") {
            HStack {
                Text("Stop Loss")
                Spacer()
                TextField("Required", text: $stopLoss)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }

            HStack {
                Text("Target 1")
                Spacer()
                TextField("Optional", text: $target1)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }

            HStack {
                Text("Target 2")
                Spacer()
                TextField("Optional", text: $target2)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }

            HStack {
                Text("Target 3")
                Spacer()
                TextField("Optional", text: $target3)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
            }
        }
    }

    // MARK: - Summary Section

    private var summarySection: some View {
        Section("Summary") {
            if entryMode == .quick {
                if let price = Double(avgPrice), let shares = Int(totalShares), shares > 0 {
                    HStack {
                        Text("Cost Basis")
                        Spacer()
                        Text(String(format: "$%.2f", price * Double(shares)))
                            .fontWeight(.semibold)
                    }
                }
            } else {
                let validEntries = entries.compactMap { entry -> (Double, Int)? in
                    guard let price = Double(entry.price),
                          let shares = Int(entry.shares),
                          shares > 0 else { return nil }
                    return (price, shares)
                }

                if !validEntries.isEmpty {
                    let totalShares = validEntries.reduce(0) { $0 + $1.1 }
                    let totalCost = validEntries.reduce(0.0) { $0 + $1.0 * Double($1.1) }
                    let avgEntry = totalCost / Double(totalShares)

                    HStack {
                        Text("Total Shares")
                        Spacer()
                        Text("\(totalShares)")
                    }

                    HStack {
                        Text("Average Entry")
                        Spacer()
                        Text(String(format: "$%.2f", avgEntry))
                    }

                    HStack {
                        Text("Cost Basis")
                        Spacer()
                        Text(String(format: "$%.2f", totalCost))
                            .fontWeight(.semibold)
                    }
                }
            }
        }
    }

    // MARK: - Validation

    private var isValid: Bool {
        if isNewPosition {
            guard let _ = Double(stopLoss) else { return false }
        }

        if entryMode == .quick {
            guard let _ = Double(avgPrice),
                  let shares = Int(totalShares),
                  shares > 0 else { return false }
            return true
        } else {
            let validEntries = entries.filter { entry in
                guard let _ = Double(entry.price),
                      let shares = Int(entry.shares),
                      shares > 0 else { return false }
                return true
            }
            return !validEntries.isEmpty
        }
    }

    // MARK: - Actions

    private func prefillFromPlan() {
        // Try to get values from existing position or trading plan
        if let position = viewModel.position {
            stopLoss = String(format: "%.2f", position.stopLoss)
            if let t1 = position.target1 { target1 = String(format: "%.2f", t1) }
            if let t2 = position.target2 { target2 = String(format: "%.2f", t2) }
            if let t3 = position.target3 { target3 = String(format: "%.2f", t3) }
        }

        // Prefill with current price
        if let price = currentPrice {
            avgPrice = String(format: "%.2f", price)
            if entries.isEmpty {
                entries = [EntryField(price: String(format: "%.2f", price), shares: "")]
            }
        }
    }

    private func submitEntry() async {
        if isNewPosition {
            // Create position first
            await viewModel.createPosition(
                stopLoss: Double(stopLoss) ?? 0,
                target1: Double(target1),
                target2: Double(target2),
                target3: Double(target3)
            )
        }

        // Add entries
        if entryMode == .quick {
            if let price = Double(avgPrice), let shares = Int(totalShares), shares > 0 {
                await viewModel.addEntry(price: price, shares: shares)
            }
        } else {
            for entry in entries {
                if let price = Double(entry.price),
                   let shares = Int(entry.shares),
                   shares > 0 {
                    await viewModel.addEntry(price: price, shares: shares)
                }
            }
        }
    }
}
