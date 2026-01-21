import Foundation

// MARK: - Admin Status

struct AdminStatusResponse: Codable {
    let isAdmin: Bool
    let userId: String
    let email: String

    enum CodingKeys: String, CodingKey {
        case isAdmin = "is_admin"
        case userId = "user_id"
        case email
    }
}

// MARK: - Usage Summary

struct UsageSummaryResponse: Codable {
    let summary: UsageSummary
    let periodDays: Int

    enum CodingKeys: String, CodingKey {
        case summary
        case periodDays = "period_days"
    }
}

struct UsageSummary: Codable {
    let userId: String?
    let periodStart: String
    let periodEnd: String

    // Claude stats
    let claudeRequests: Int
    let claudeInputTokens: Int
    let claudeOutputTokens: Int
    let claudeCost: Double

    // Grok stats
    let grokRequests: Int
    let grokInputTokens: Int
    let grokOutputTokens: Int
    let grokCost: Double
    let grokToolCalls: Int
    let grokToolCost: Double

    // Totals
    let totalRequests: Int
    let totalTokens: Int
    let totalCost: Double

    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case periodStart = "period_start"
        case periodEnd = "period_end"
        case claudeRequests = "claude_requests"
        case claudeInputTokens = "claude_input_tokens"
        case claudeOutputTokens = "claude_output_tokens"
        case claudeCost = "claude_cost"
        case grokRequests = "grok_requests"
        case grokInputTokens = "grok_input_tokens"
        case grokOutputTokens = "grok_output_tokens"
        case grokCost = "grok_cost"
        case grokToolCalls = "grok_tool_calls"
        case grokToolCost = "grok_tool_cost"
        case totalRequests = "total_requests"
        case totalTokens = "total_tokens"
        case totalCost = "total_cost"
    }
}

// MARK: - Usage By User

struct AllUsersSummaryResponse: Codable {
    let users: [UserUsageSummary]
    let totalCount: Int
    let periodStart: String
    let periodEnd: String
    let grandTotalCost: Double

    enum CodingKeys: String, CodingKey {
        case users
        case totalCount = "total_count"
        case periodStart = "period_start"
        case periodEnd = "period_end"
        case grandTotalCost = "grand_total_cost"
    }
}

struct UserUsageSummary: Codable, Identifiable {
    let userId: String
    let email: String?
    let totalRequests: Int
    let totalTokens: Int
    let totalCost: Double

    // Provider breakdown
    let claudeCost: Double
    let grokCost: Double

    // Activity
    let lastRequestAt: String?

    // Operation counts
    let planGenerations: Int
    let chatRequests: Int
    let evaluations: Int
    let orchestratorCalls: Int
    let subagentCalls: Int
    let imageAnalyses: Int

    // Operation costs
    let planGenerationCost: Double
    let chatCost: Double
    let evaluationCost: Double
    let orchestratorCost: Double
    let subagentCost: Double
    let imageAnalysisCost: Double

    var id: String { userId }

    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case email
        case totalRequests = "total_requests"
        case totalTokens = "total_tokens"
        case totalCost = "total_cost"
        case claudeCost = "claude_cost"
        case grokCost = "grok_cost"
        case lastRequestAt = "last_request_at"
        case planGenerations = "plan_generations"
        case chatRequests = "chat_requests"
        case evaluations
        case orchestratorCalls = "orchestrator_calls"
        case subagentCalls = "subagent_calls"
        case imageAnalyses = "image_analyses"
        case planGenerationCost = "plan_generation_cost"
        case chatCost = "chat_cost"
        case evaluationCost = "evaluation_cost"
        case orchestratorCost = "orchestrator_cost"
        case subagentCost = "subagent_cost"
        case imageAnalysisCost = "image_analysis_cost"
    }

    /// Display name (email or truncated user ID)
    var displayName: String {
        if let email = email, !email.isEmpty {
            return email.components(separatedBy: "@").first ?? email
        }
        return String(userId.prefix(8)) + "..."
    }
}

// MARK: - Daily Costs

struct DailyCostsResponse: Codable {
    let dailyCosts: [DailyCostItem]
    let totalCost: Double
    let periodDays: Int

    enum CodingKeys: String, CodingKey {
        case dailyCosts = "daily_costs"
        case totalCost = "total_cost"
        case periodDays = "period_days"
    }
}

struct DailyCostItem: Codable, Identifiable {
    let date: String
    let requests: Int
    let tokens: Int
    let cost: Double
    let claudeCost: Double
    let grokCost: Double

    var id: String { date }

    enum CodingKeys: String, CodingKey {
        case date
        case requests
        case tokens
        case cost
        case claudeCost = "claude_cost"
        case grokCost = "grok_cost"
    }

    /// Parsed date for chart display
    var parsedDate: Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: date)
    }

    /// Short date label (e.g., "Jan 15")
    var shortDateLabel: String {
        guard let date = parsedDate else { return self.date }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: date)
    }
}

// MARK: - Usage By Operation

struct UsageByOperationResponse: Codable {
    let userId: String?
    let periodStart: String
    let periodEnd: String
    let breakdowns: [OperationTypeBreakdown]
    let totalCost: Double

    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case periodStart = "period_start"
        case periodEnd = "period_end"
        case breakdowns
        case totalCost = "total_cost"
    }
}

struct OperationTypeBreakdown: Codable, Identifiable {
    let operationType: String
    let requestCount: Int
    let inputTokens: Int
    let outputTokens: Int
    let totalTokens: Int
    let totalCost: Double
    let avgCostPerRequest: Double

    var id: String { operationType }

    enum CodingKeys: String, CodingKey {
        case operationType = "operation_type"
        case requestCount = "request_count"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case totalTokens = "total_tokens"
        case totalCost = "total_cost"
        case avgCostPerRequest = "avg_cost_per_request"
    }

    /// Human-readable operation name
    var displayName: String {
        switch operationType {
        case "plan_generation": return "Plan Generation"
        case "plan_evaluation": return "Plan Evaluation"
        case "chat": return "Chat"
        case "orchestrator": return "Orchestrator"
        case "subagent": return "Sub-agents"
        case "image_analysis": return "Vision Analysis"
        default: return operationType.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    /// Icon for operation type
    var icon: String {
        switch operationType {
        case "plan_generation": return "doc.text.fill"
        case "plan_evaluation": return "checkmark.circle.fill"
        case "chat": return "bubble.left.and.bubble.right.fill"
        case "orchestrator": return "cpu.fill"
        case "subagent": return "person.3.fill"
        case "image_analysis": return "eye.fill"
        default: return "circle.fill"
        }
    }

    /// Color for operation type
    var color: String {
        switch operationType {
        case "plan_generation": return "blue"
        case "plan_evaluation": return "green"
        case "chat": return "purple"
        case "orchestrator": return "orange"
        case "subagent": return "cyan"
        case "image_analysis": return "pink"
        default: return "gray"
        }
    }
}
