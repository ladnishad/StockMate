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

    // MARK: - Computed Properties

    var accessToken: String? {
        keychain.accessToken
    }

    var userId: String? {
        currentUser?.id ?? keychain.userId
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
        }
    }

    // MARK: - Public Methods

    /// Check if existing session is valid by calling /auth/me
    /// If access token is expired, attempts to refresh it before logging out
    func checkSession() async {
        guard keychain.hasValidSession else {
            isAuthenticated = false
            currentUser = nil
            return
        }

        do {
            let user = try await fetchCurrentUser()
            currentUser = user
            isAuthenticated = true
        } catch {
            // Token might be expired - try refreshing before giving up
            print("Session check failed: \(error), attempting token refresh...")

            do {
                try await refreshAccessToken()
                print("Token refresh successful, retrying session check...")

                // Retry fetching user with new token
                let user = try await fetchCurrentUser()
                currentUser = user
                isAuthenticated = true
                print("Session restored successfully")
            } catch {
                // Refresh also failed - now we can logout
                print("Token refresh failed: \(error), logging out")
                await logout()
            }
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
            throw AuthError.noRefreshToken
        }

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
        } else {
            // Refresh failed, need to re-login
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
