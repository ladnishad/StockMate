"""US Market Holidays and Trading Hours.

This module provides fallback data for market hours when the Alpaca API
is unavailable. The primary source of truth should be the Alpaca clock
endpoint - this data is only used as a backup.

Holiday data sources:
- NYSE official calendar: https://www.nyse.com/trade/hours-calendars
- NASDAQ follows the same schedule as NYSE for US equities

IMPORTANT: This data requires annual updates. The Alpaca clock API should
be used as the primary source to avoid maintenance burden.
"""

from datetime import time

# Regular market hours (US Eastern Time)
MARKET_OPEN_TIME = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET
EARLY_CLOSE_TIME = time(13, 0)   # 1:00 PM ET (early close days)

# US Market Holidays (NYSE/NASDAQ) - Full day closures
# Format: "YYYY-MM-DD"
#
# Annual US market holidays:
# - New Year's Day (January 1, or observed Friday/Monday if weekend)
# - Martin Luther King Jr. Day (third Monday in January)
# - Washington's Birthday/Presidents Day (third Monday in February)
# - Good Friday (Friday before Easter Sunday - varies each year)
# - Memorial Day (last Monday in May)
# - Juneteenth National Independence Day (June 19, or observed Friday/Monday)
# - Independence Day (July 4, or observed Friday/Monday if weekend)
# - Labor Day (first Monday in September)
# - Thanksgiving Day (fourth Thursday in November)
# - Christmas Day (December 25, or observed Friday/Monday if weekend)

MARKET_HOLIDAYS = {
    # 2024 Holidays
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas

    # 2025 Holidays
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas

    # 2026 Holidays (Source: NYSE official calendar)
    "2026-01-01",  # New Year's Day (Thursday)
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth (Friday)
    "2026-07-03",  # Independence Day (observed - July 4 is Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas (Friday)

    # 2027 Holidays (calculated based on standard rules)
    "2027-01-01",  # New Year's Day (Friday)
    "2027-01-18",  # Martin Luther King Jr. Day
    "2027-02-15",  # Presidents Day
    "2027-03-26",  # Good Friday (Easter is March 28)
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observed - June 19 is Saturday)
    "2027-07-05",  # Independence Day (observed - July 4 is Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed - December 25 is Saturday)

    # 2028 Holidays (calculated based on standard rules)
    "2028-01-17",  # Martin Luther King Jr. Day
    "2028-02-21",  # Presidents Day
    "2028-04-14",  # Good Friday (Easter is April 16)
    "2028-05-29",  # Memorial Day
    "2028-06-19",  # Juneteenth (Monday)
    "2028-07-04",  # Independence Day (Tuesday)
    "2028-09-04",  # Labor Day
    "2028-11-23",  # Thanksgiving
    "2028-12-25",  # Christmas (Monday)
}

# Early close days (1:00 PM ET close)
# Typically the day after Thanksgiving and Christmas Eve
# Also sometimes day before Independence Day if it falls mid-week

EARLY_CLOSE_DAYS = {
    # 2024 Early Closes
    "2024-07-03",  # Day before Independence Day
    "2024-11-29",  # Day after Thanksgiving
    "2024-12-24",  # Christmas Eve

    # 2025 Early Closes
    "2025-07-03",  # Day before Independence Day (July 4 is Friday)
    "2025-11-28",  # Day after Thanksgiving
    "2025-12-24",  # Christmas Eve

    # 2026 Early Closes
    # Note: July 3 is a full closure (Independence Day observed)
    "2026-11-27",  # Day after Thanksgiving
    "2026-12-24",  # Christmas Eve

    # 2027 Early Closes
    "2027-07-02",  # Day before Independence Day (observed Monday)
    "2027-11-26",  # Day after Thanksgiving
    "2027-12-23",  # Day before Christmas (observed on Dec 24)

    # 2028 Early Closes
    "2028-07-03",  # Day before Independence Day
    "2028-11-24",  # Day after Thanksgiving
    "2028-12-22",  # Friday before Christmas (Dec 25 is Monday)
}


def is_market_holiday(date_str: str) -> bool:
    """Check if a date is a market holiday.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        True if the date is a market holiday
    """
    return date_str in MARKET_HOLIDAYS


def is_early_close_day(date_str: str) -> bool:
    """Check if a date is an early close day.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        True if the date is an early close day (1:00 PM close)
    """
    return date_str in EARLY_CLOSE_DAYS


def get_market_close_time_for_date(date_str: str) -> time:
    """Get the market close time for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Market close time (1:00 PM for early close, 4:00 PM otherwise)
    """
    if is_early_close_day(date_str):
        return EARLY_CLOSE_TIME
    return MARKET_CLOSE_TIME


def get_years_covered() -> list:
    """Get the years covered by the holiday data.

    Returns:
        List of years (as integers) covered by the holiday data
    """
    years = set()
    for date_str in MARKET_HOLIDAYS:
        years.add(int(date_str[:4]))
    return sorted(years)


def is_year_covered(year: int) -> bool:
    """Check if a year has holiday data.

    Args:
        year: The year to check

    Returns:
        True if holiday data exists for the year
    """
    return year in get_years_covered()
