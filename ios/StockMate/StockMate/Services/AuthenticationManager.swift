//
//  AuthenticationManager.swift
//  StockMate
//
//  Manages authentication state, token storage, and auth API calls.
//

import Foundation
import SwiftUI
import Combine

@MainActor
final class AuthenticationManager: ObservableObject {

    // MARK: - Singleton

    static let shared = AuthenticationManager()

    // MARK: - Published Properties

    @Published var isAuthenticated: Bool = false
    @Published var currentUser: AuthUser?
    @Published var isLoading: Bool = false
    @Published var error: String?

    // MARK: - Private Properties

    private let keychain = KeychainHelper.shared
    private let baseURL = "https://stockmate-fggr.onrender.com"

    /// Timer for proactive token refresh
    private var tokenRefreshTimer: Timer?

    /// Minimum time before expiration to trigger proactive refresh (5 minutes)
    private let proactiveRefreshThreshold: TimeInterval = 5 * 60

    // MARK: - Computed Properties

    var accessToken: String? {
        keychain.accessToken
    }

    var userId: String? {
        currentUser?.id ?? keychain.userId
    }

    /// Check if the access token is expiring soon (within 5 minutes)
    /// This gives us enough buffer to refresh before actual expiration
    var isTokenExpiringSoon: Bool {
        guard let expirationDate = keychain.tokenExpirationDate else {
            // No expiration stored, assume it might be expired
            return true
        }
        // Consider token "expiring soon" if less than 5 minutes remaining
        return expirationDate.timeIntervalSinceNow < proactiveRefreshThreshold
    }

    /// Check if the access token has already expired
    var isTokenExpired: Bool {
        guard let expirationDate = keychain.tokenExpirationDate else {
            return true
        }
        return expirationDate.timeIntervalSinceNow <= 0
    }

    /// Time remaining until token expires (negative if already expired)
    var tokenTimeRemaining: TimeInterval {
        keychain.tokenExpirationDate?.timeIntervalSinceNow ?? -1
    }

    // MARK: - Init

    private init() {
        // Check for existing session on init
        if keychain.hasValidSession {
            isAuthenticated = true
            // Restore user info from keychain
            if let email = keychain.userEmail, let id = keychain.userId {
                currentUser = AuthUser(id: id, email: email, emailVerified: true, createdAt: nil)
            }
            // Schedule token refresh if we have a valid session
            scheduleTokenRefresh()
        }
    }

    // MARK: - Token Refresh Scheduling

    /// Schedule a timer to refresh the token before it expires
    private func scheduleTokenRefresh() {
        // Cancel any existing timer
        tokenRefreshTimer?.invalidate()
        tokenRefreshTimer = nil

        guard let expirationDate = keychain.tokenExpirationDate else {
            print("No token expiration date stored, cannot schedule refresh")
            return
        }

        // Calculate when to refresh (5 minutes before expiration)
        let refreshTime = expirationDate.addingTimeInterval(-proactiveRefreshThreshold)
        let timeUntilRefresh = refreshTime.timeIntervalSinceNow

        if timeUntilRefresh <= 0 {
            // Token is already expiring soon or expired, refresh immediately
            print("Token expiring soon or expired, refreshing immediately")
            Task {
                try? await refreshAccessToken()
            }
            return
        }

        print("Scheduling token refresh in \(Int(timeUntilRefresh)) seconds")

        // Schedule the timer
        tokenRefreshTimer = Timer.scheduledTimer(withTimeInterval: timeUntilRefresh, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                print("Proactive token refresh triggered by timer")
                do {
                    try await self.refreshAccessToken()
                    print("Proactive token refresh successful")
                } catch {
                    print("Proactive token refresh failed: \(error)")
                    // Don't logout here - the next API call will handle it
                }
            }
        }
    }

    /// Cancel the token refresh timer
    private func cancelTokenRefreshTimer() {
        tokenRefreshTimer?.invalidate()
        tokenRefreshTimer = nil
    }

    // MARK: - Public Methods

    /// Check if existing session is valid by calling /auth/me
    /// If access token is expired, attempts to refresh it before logging out
    /// Distinguishes between network errors (keep session) and auth errors (logout)
    func checkSession() async {
        guard keychain.hasValidSession else {
            print("[Auth] No valid session in keychain")
            isAuthenticated = false
            currentUser = nil
            cancelTokenRefreshTimer()
            return
        }

        // Log token status for debugging
        let timeRemaining = tokenTimeRemaining
        if timeRemaining > 0 {
            print("[Auth] Token valid, expires in \(Int(timeRemaining)) seconds (\(Int(timeRemaining / 60)) minutes)")
        } else {
            print("[Auth] Token expired \(Int(-timeRemaining)) seconds ago")
        }

        // Log last refresh time for debugging
        if let lastRefresh = keychain.lastRefreshTime {
            let timeSinceRefresh = Date().timeIntervalSince(lastRefresh)
            print("[Auth] Last token refresh was \(Int(timeSinceRefresh / 60)) minutes ago")
        }

        // Proactively refresh token if expiring soon or already expired
        if isTokenExpiringSoon {
            let status = isTokenExpired ? "expired" : "expiring soon"
            print("[Auth] Token \(status), proactively refreshing...")
            do {
                try await refreshAccessToken()
                print("[Auth] Proactive token refresh successful")
            } catch {
                print("[Auth] Proactive refresh failed: \(error)")
                // Continue to try fetching user anyway - maybe the token is still valid
            }
        }

        do {
            let user = try await fetchCurrentUser()
            currentUser = user
            isAuthenticated = true
            // Reschedule token refresh timer
            scheduleTokenRefresh()
            print("[Auth] Session valid, user: \(user.email)")
        } catch let authError as AuthError {
            // Handle specific auth errors
            switch authError {
            case .notAuthenticated:
                // 401 error - try refresh
                print("[Auth] Session check got 401, attempting token refresh...")
                await handleTokenRefresh()

            case .invalidResponse:
                // Could be network issue - don't logout, keep existing session
                print("[Auth] Invalid response during session check - possible network issue, keeping session")
                // Keep isAuthenticated = true since we have valid keychain session

            default:
                // Other auth errors - try refresh as fallback
                print("[Auth] Session check failed with \(authError), attempting refresh...")
                await handleTokenRefresh()
            }
        } catch let urlError as URLError {
            // Network errors - don't logout, just log
            print("[Auth] Network error during session check: \(urlError.localizedDescription)")
            print("[Auth] Keeping session active, will retry on next app foreground")
            // Keep isAuthenticated = true since we have valid session in keychain
        } catch {
            // Other unexpected errors - don't logout on unknown errors
            print("[Auth] Unexpected error during session check: \(error)")
            // Keep session active to avoid false logouts
        }
    }

    /// Handle token refresh with proper error handling
    private func handleTokenRefresh() async {
        do {
            try await refreshAccessToken()
            print("[Auth] Token refresh successful, retrying session check...")

            // Retry fetching user with new token
            let user = try await fetchCurrentUser()
            currentUser = user
            isAuthenticated = true
            // Schedule next token refresh
            scheduleTokenRefresh()
            print("[Auth] Session restored successfully")
        } catch let authError as AuthError {
            switch authError {
            case .noRefreshToken:
                // No refresh token available - must login again
                print("[Auth] No refresh token available, logging out")
                await logout()
            case .refreshFailed:
                // Refresh token is invalid or expired - must login again
                // This typically happens when the user hasn't used the app
                // for longer than the refresh token lifetime (e.g., 7-30 days)
                print("[Auth] Refresh token expired or invalid, logging out")
                print("[Auth] User will need to login again - this is expected after extended inactivity")
                await logout()
            default:
                // Other auth errors during refresh - might be temporary
                print("[Auth] Refresh encountered error: \(authError), keeping session for retry")
            }
        } catch let urlError as URLError {
            // Network error during refresh - don't logout
            print("[Auth] Network error during token refresh: \(urlError.localizedDescription)")
            print("[Auth] Keeping session, will retry later")
        } catch {
            // Unknown errors - log but don't logout
            print("[Auth] Unexpected error during refresh: \(error), keeping session")
        }
    }

    /// Login with email and password
    func login(email: String, password: String) async throws {
        isLoading = true
        self.error = nil

        defer { isLoading = false }

        let url = URL(string: "\(baseURL)/auth/login")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = LoginRequest(email: email, password: password)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.invalidResponse
        }

        if httpResponse.statusCode == 200 || httpResponse.statusCode == 201 {
            let decoder = JSONDecoder()
            let authResponse = try decoder.decode(AuthResponse.self, from: data)

            // Save to keychain
            keychain.saveAuthSession(response: authResponse)

            // Update state
            currentUser = authResponse.user
            isAuthenticated = true

            // Schedule proactive token refresh
            scheduleTokenRefresh()
            print("[Auth] Login successful, token expires in \(authResponse.expiresIn) seconds")
        } else if httpResponse.statusCode == 401 {
            throw AuthError.invalidCredentials
        } else {
            // Try to parse error message
            if let errorResponse = try? JSONDecoder().decode(AuthErrorResponse.self, from: data) {
                throw AuthError.serverError(errorResponse.detail)
            }
            throw AuthError.httpError(httpResponse.statusCode)
        }
    }

    /// Sign up with email and password
    func signup(email: String, password: String) async throws {
        isLoading = true
        self.error = nil

        defer { isLoading = false }

        let url = URL(string: "\(baseURL)/auth/signup")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = SignupRequest(email: email, password: password)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.invalidResponse
        }

        if httpResponse.statusCode == 200 || httpResponse.statusCode == 201 {
            // Check if we got a full auth response or just a message (email confirmation required)
            if let authResponse = try? JSONDecoder().decode(AuthResponse.self, from: data) {
                // Direct login (no email confirmation required)
                keychain.saveAuthSession(response: authResponse)
                currentUser = authResponse.user
                isAuthenticated = true
                // Schedule proactive token refresh
                scheduleTokenRefresh()
                print("[Auth] Signup successful, token expires in \(authResponse.expiresIn) seconds")
            } else if let _ = try? JSONDecoder().decode(AuthErrorResponse.self, from: data) {
                // Email confirmation required - throw specific error
                throw AuthError.emailConfirmationRequired
            }
        } else if httpResponse.statusCode == 409 {
            throw AuthError.emailAlreadyExists
        } else {
            if let errorResponse = try? JSONDecoder().decode(AuthErrorResponse.self, from: data) {
                throw AuthError.serverError(errorResponse.detail)
            }
            throw AuthError.httpError(httpResponse.statusCode)
        }
    }

    /// Logout - clear tokens and state
    func logout() async {
        print("[Auth] Logging out user")

        // Cancel token refresh timer
        cancelTokenRefreshTimer()

        // Optionally call server logout
        if let token = keychain.accessToken {
            let url = URL(string: "\(baseURL)/auth/logout")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

            // Fire and forget - don't wait for response
            Task {
                try? await URLSession.shared.data(for: request)
            }
        }

        // Clear local state
        keychain.clearAll()
        currentUser = nil
        isAuthenticated = false
        print("[Auth] Logout complete, all tokens cleared")
    }

    /// Request password reset email
    func resetPassword(email: String) async throws {
        isLoading = true
        self.error = nil

        defer { isLoading = false }

        let url = URL(string: "\(baseURL)/auth/reset-password")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = PasswordResetRequest(email: email)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.invalidResponse
        }

        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(AuthErrorResponse.self, from: data) {
                throw AuthError.serverError(errorResponse.detail)
            }
            throw AuthError.httpError(httpResponse.statusCode)
        }
    }

    /// Refresh access token using refresh token
    func refreshAccessToken() async throws {
        guard let refreshToken = keychain.refreshToken else {
            print("[Auth] No refresh token available in keychain")
            throw AuthError.noRefreshToken
        }

        print("[Auth] Attempting to refresh access token...")

        let url = URL(string: "\(baseURL)/auth/refresh")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = RefreshTokenRequest(refreshToken: refreshToken)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.invalidResponse
        }

        if httpResponse.statusCode == 200 {
            let authResponse = try JSONDecoder().decode(AuthResponse.self, from: data)
            keychain.saveAuthSession(response: authResponse)
            currentUser = authResponse.user
            // Schedule next proactive refresh
            scheduleTokenRefresh()
            print("[Auth] Token refresh successful, new token expires in \(authResponse.expiresIn) seconds")
        } else {
            // Refresh failed, need to re-login
            // This typically means the refresh token has expired
            print("[Auth] Token refresh failed with status \(httpResponse.statusCode)")
            if let errorBody = String(data: data, encoding: .utf8) {
                print("[Auth] Error response: \(errorBody)")
            }
            throw AuthError.refreshFailed
        }
    }

    // MARK: - Private Methods

    private func fetchCurrentUser() async throws -> AuthUser {
        guard let token = keychain.accessToken else {
            throw AuthError.notAuthenticated
        }

        let url = URL(string: "\(baseURL)/auth/me")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthError.invalidResponse
        }

        if httpResponse.statusCode == 200 {
            // The /auth/me endpoint returns a UserResponse which is slightly different
            struct UserResponse: Codable {
                let id: String
                let email: String
                let emailVerified: Bool
                let createdAt: String?

                enum CodingKeys: String, CodingKey {
                    case id
                    case email
                    case emailVerified = "email_verified"
                    case createdAt = "created_at"
                }
            }

            let userResponse = try JSONDecoder().decode(UserResponse.self, from: data)
            return AuthUser(
                id: userResponse.id,
                email: userResponse.email,
                emailVerified: userResponse.emailVerified,
                createdAt: userResponse.createdAt
            )
        } else if httpResponse.statusCode == 401 {
            throw AuthError.notAuthenticated
        } else {
            throw AuthError.httpError(httpResponse.statusCode)
        }
    }
}

// MARK: - Auth Errors

enum AuthError: LocalizedError {
    case invalidCredentials
    case emailAlreadyExists
    case emailConfirmationRequired
    case invalidResponse
    case httpError(Int)
    case serverError(String)
    case notAuthenticated
    case noRefreshToken
    case refreshFailed
    case networkError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidCredentials:
            return "Invalid email or password"
        case .emailAlreadyExists:
            return "An account with this email already exists"
        case .emailConfirmationRequired:
            return "Please check your email to confirm your account"
        case .invalidResponse:
            return "Invalid server response"
        case .httpError(let code):
            return "Server error (code: \(code))"
        case .serverError(let message):
            return message
        case .notAuthenticated:
            return "Not authenticated"
        case .noRefreshToken:
            return "No refresh token available"
        case .refreshFailed:
            return "Session expired. Please login again"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}
