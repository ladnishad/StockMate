"""Admin module for StockMate.

Provides admin-only endpoints for usage tracking, user management, and system monitoring.
"""

from app.admin.router import router as admin_router

__all__ = ["admin_router"]
