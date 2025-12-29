//
//  AuthModels.swift
//  StockMate
//
//  Authentication models for Supabase auth integration.
//

import Foundation

// MARK: - User Model

struct AuthUser: Codable, Identifiable {
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

// MARK: - Auth Response

struct AuthResponse: Codable {
    let accessToken: String
    let refreshToken: String
    let tokenType: String
    let expiresIn: Int
    let user: AuthUser

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case tokenType = "token_type"
        case expiresIn = "expires_in"
        case user
    }
}

// MARK: - Auth Requests

struct LoginRequest: Codable {
    let email: String
    let password: String
}

struct SignupRequest: Codable {
    let email: String
    let password: String
}

struct RefreshTokenRequest: Codable {
    let refreshToken: String

    enum CodingKeys: String, CodingKey {
        case refreshToken = "refresh_token"
    }
}

struct PasswordResetRequest: Codable {
    let email: String
}

// MARK: - Auth Error Response

struct AuthErrorResponse: Codable {
    let detail: String
}

// MARK: - Password Reset Response

struct PasswordResetResponse: Codable {
    let message: String
}

// MARK: - Message Response

struct MessageResponse: Codable {
    let message: String
}
