# Y Combinator Application - StockMate

## Company Basics

### 1. Describe what your company does in 50 characters or less.

AI copilot that gives retail traders pro-level edge

---

### 2. What is your company going to make? Please describe your product and what it does or will do.

StockMate is an AI-powered stock analysis platform that brings institutional-grade trading tools to everyday investors — think Cursor, but for stock trading. Users can choose between AI models like Claude and Grok to analyze stocks across multiple timeframes, detect chart patterns, calculate volume profiles, and generate specific trade plans with entry prices, stop losses, and profit targets — all personalized to their trading style.

Our platform runs a 15-factor weighted scoring system (VWAP, RSI, MACD, volume profile, divergences, Fibonacci, chart patterns, and more) that gives traders a clear confidence score on whether to buy, removing guesswork and emotional decision-making. The AI agent continuously monitors watchlists and alerts users when price approaches key support/resistance levels, when breakouts occur, or when stop losses are at risk.

We serve four distinct trader profiles — day traders, swing traders, position traders, and long-term investors — each with customized timeframes, risk parameters, and indicator weightings. The iOS app delivers all of this in a conversational chat interface where you can ask the AI about any stock and get back a professional-grade trade plan in seconds, something that takes even experienced traders hours to build manually.

---

### 3. Where do you live now, and where would the company be based after YC?

[FILL IN: Your current city/state and where you'd base the company — San Francisco is typical post-YC]

---

### 4. How far along are you?

We have a fully functional product deployed in production. The backend is built with FastAPI and deployed on Render.com with Docker. The iOS app is built natively in SwiftUI and is app-store ready. We have real-time market data streaming via Alpaca's SIP feed, dual AI model integration (Claude and Grok), and a complete four-tier subscription model (Free, $20/mo Premium, $50/mo Pro, $200/mo Unlimited). Our admin dashboard tracks per-user usage, API costs by provider, and operation breakdowns. The technical analysis engine includes 34 LLM-ready tool functions spanning market data, technical indicators, analysis, and market scanning. We are ready to launch on the App Store.

[FILL IN: Add current user count, any beta testers, waitlist numbers, revenue if any]

---

### 5. How long have each of you been working on this? How much of that has been full-time?

[FILL IN: Timeline for each founder — e.g., "I've been working on StockMate for X months, Y of which full-time. My co-founder joined X weeks ago."]

---

### 6. If you have a demo, what's the url?

[FILL IN: Demo URL, TestFlight link, or API docs URL at your Render deployment]

---

## Founder / Team Details

### 7. Who writes code, or does other technical work on your product? Was any of it done by a non-founder?

All code has been written by the founders. The backend (FastAPI, Python), AI agent system, technical analysis engine, and iOS app (SwiftUI) were built entirely in-house. No outside contractors or agencies were used. We own 100% of the IP.

[FILL IN: Specify which founder built which parts]

---

### 8. How long have the founders known one another and how did you meet?

[FILL IN: Your story — how you met, how long you've known each other]

---

### 9. Please tell us about an interesting project, preferably outside of class or work, that two or more of you created together.

[FILL IN: A project you and your co-founder built together — ideally something that shows technical ability, scrappiness, and complementary skills]

---

### 10. Please tell us in one or two sentences about something impressive that each founder has built or achieved.

[FILL IN: This is YC's most important question. For each founder, one genuinely impressive achievement. Be specific with numbers/outcomes. Examples: "Built X that got Y users," "Grew Z from 0 to N," "Won competition X against N teams"]

---

### 11. Please tell us about the time you most successfully hacked some (non-computer) system to your advantage.

[FILL IN: A story showing resourcefulness and willingness to bend rules creatively — YC loves founders who find unconventional shortcuts]

---

## Product & Market

### 12. What's new about what you're making? What substitutes do people resort to because it doesn't exist yet (or they don't know about it)?

Today, retail traders cobble together a fragmented stack: TradingView for charts ($15-60/mo), a separate screener like Finviz, Twitter/X for sentiment, YouTube for pattern analysis, and a spreadsheet to track positions. Even with all that, they're still guessing on entries, exits, and position sizes. Professional traders at hedge funds have Bloomberg terminals ($24,000/year) and teams of analysts running the same analysis our AI does in seconds.

What's new is the AI copilot approach. No existing platform uses LLMs as an intelligent layer on top of professional-grade technical analysis. StockMate doesn't just show you charts — it runs 15 weighted indicators across multiple timeframes, detects divergences and chart patterns, calculates volume profiles for institutional positioning, and then synthesizes all of that into a specific trade plan with exact entry, stop loss, and targets calibrated to your risk tolerance and trading style. Users can choose between Claude and Grok models, each with different strengths — Claude for deep analysis, Grok for real-time X/Twitter sentiment integration. The closest substitute is paying a human trading mentor $200-500/month who gives you worse analysis with multi-hour delays.

---

### 13. Who are your competitors? What do you understand about your business that they don't?

**Direct competitors:** TradingView (charting), Seeking Alpha (analysis), Stock Advisor by Motley Fool (recommendations), Trade Ideas (scanning).

**What they miss:** These platforms are all tools-first — they give you the instruments but expect you to be the expert. TradingView is powerful but overwhelming for non-professionals. Motley Fool gives buy recommendations without specific entries, stops, or position sizing. None of them use AI as an intelligent analyst that personalizes analysis to your trading style and risk profile.

What we understand: Retail traders don't need more tools — they need a co-pilot that synthesizes tools for them. The future of trading platforms isn't showing more data, it's showing the right decision. Our 15-factor scoring system with per-profile confidence thresholds (70% for day traders, 60% for long-term investors) means we're not just saying "buy AAPL" — we're saying "buy AAPL at $182.50, stop at $178.20 (1.5x ATR), target $191.30, risk 1% of your account, allocate 847 shares." That specificity is what separates us.

---

### 14. How do or will you make money? How much could you make?

**Revenue model:** Four-tier monthly SaaS subscription.

- **Base (Free):** 2 watchlist stocks, Claude AI, basic plans — designed to convert users
- **Premium ($20/mo):** 5 stocks, multi-model AI (Claude + Grok), X/Twitter sentiment
- **Pro ($50/mo):** 20 stocks, professional-grade analysis, priority support
- **Unlimited ($200/mo):** Unlimited watchlist, all features, dedicated support

**Unit economics:** Our primary costs are AI API calls and market data. We track per-user API costs in real-time through our admin dashboard, broken down by provider and operation type. Claude analysis costs roughly $0.01-0.05 per analysis depending on depth, meaning even at the $20/mo tier our margins are strong.

**Market size:** There are 60+ million active retail traders in the US alone (up from 10 million pre-2020). The retail trading tools market is projected at $12B+ by 2028. If we capture just 0.1% of US retail traders at an average $35/mo ARPU, that's $25M ARR. The real upside is international expansion — retail trading is exploding in India, Southeast Asia, and Latin America.

---

### 15. How do users find your product? How did you get the users you have now?

[FILL IN: Your actual distribution strategy. Some honest options:]

Our initial distribution strategy targets three channels:

1. **Trading communities:** Reddit (r/wallstreetbets has 16M members, r/stocks has 7M), StockTwits, and Discord trading servers. Traders actively share tools that give them an edge — our AI-generated trade plans with specific entries and stops are inherently shareable content.

2. **Content marketing:** We'll publish AI-generated analysis on trending stocks (earnings plays, breakout setups) on X/Twitter and YouTube. Each analysis is a product demo — "Here's what StockMate's AI found on NVDA before earnings."

3. **Word of mouth from results:** When a trade plan works, traders tell other traders. Our structured plans with exact numbers create verifiable track records that drive organic growth.

[FILL IN: Add how you actually got your current users — beta invites, friends, ProductHunt, etc.]

---

## Users & Metrics

### 16. How many active users or customers do you have?

[FILL IN: Honest number — even if it's small. "5 beta users" is fine. YC has funded pre-launch companies.]

---

### 17. Do you have revenue?

[FILL IN: Yes/No and numbers if yes. Pre-revenue is fine — say so honestly.]

---

## Financials

### 18. How much money do you spend per month?

Our current infrastructure costs are minimal:
- Render.com hosting: $7/month
- Alpaca market data (IEX feed): Free (SIP feed for production: $99/month)
- AI API costs: Variable based on usage (tracked per-user in our admin dashboard)
- Apple Developer Program: $99/year
- Supabase (auth + database): Free tier

[FILL IN: Total monthly burn including any salaries, other costs]

---

### 19. How much money does your company have in the bank now?

[FILL IN: Honest number]

---

### 20. How long is your runway?

[FILL IN: Based on burn rate and bank balance]

---

## Legal & Equity

### 21. Have you formed ANY legal entity yet?

[FILL IN: Yes/No — if yes, specify type (e.g., Delaware C Corp). If no, that's fine — many YC companies incorporate during the batch.]

---

### 22. Please describe the breakdown of the equity ownership in percentages among the founders, employees and any other stockholders.

[FILL IN: e.g., "Founder 1 (CEO): 60%, Founder 2 (CTO): 40%" — YC expects founders to have at least 10% each]

---

### 23. Have you taken any investment yet?

[FILL IN: Yes/No — if bootstrapped, say so. It's a strength.]

---

## Other

### 24. Is there anything else we should know about your company?

A few things that don't fit neatly into other answers:

**We built the hard thing first.** Our technical analysis engine has 34 production-ready functions — divergence detection, volume profile with VPOC/HVN analysis, chart pattern recognition (head & shoulders, double tops/bottoms, triangles, flags), multi-timeframe confluence scoring, and structural support/resistance with 22 detection methods. This isn't a ChatGPT wrapper that asks an LLM to "analyze AAPL." The AI orchestrates professional-grade quantitative tools, and the LLM synthesizes the structured output into actionable plans.

**We're model-agnostic by design.** Users can choose between Claude and Grok today, and we can add new models as they emerge. This is a deliberate architectural choice — just like Cursor lets developers pick their preferred model, we let traders pick theirs. Grok uniquely offers real-time X/Twitter sentiment search, which is valuable for momentum traders. This multi-model approach also gives us negotiating leverage on API pricing as we scale.

**Our cost structure scales well.** We track per-user, per-operation API costs in real-time. We know exactly what a plan generation costs vs. a quick chat vs. a plan evaluation. This lets us price tiers accurately and maintain margins as we grow.

---

### 25. If you had any other ideas you considered applying with, please list them.

[FILL IN: List 1-3 other ideas you've considered — YC sometimes funds teams based on alternate ideas]

---

### 26. Please tell us something surprising or amusing that one of you has discovered.

[FILL IN: Something genuinely interesting — could be a market insight, a technical discovery, or a life observation. Make it memorable.]

---

### 27. What convinced you to apply to Y Combinator?

[FILL IN: Be genuine. Some angles that work:]

The fintech companies that came out of YC — Stripe, Brex, Coinbase — succeeded because YC helped them navigate the unique regulatory and distribution challenges of financial products. We're building in a space where trust matters enormously (people's money is on the line), and YC's brand gives us instant credibility with users who are understandably skeptical of new trading tools. Beyond brand, we need help with our go-to-market — we've built a strong product but haven't cracked distribution yet, and YC's network of founders who've solved similar cold-start problems is exactly what we need.

[FILL IN: Add personal connection — did a YC founder encourage you? Did you attend a YC event?]

---

## Founder Video Script (1 minute)

*Note: This is a suggested script for your 1-minute unlisted YouTube video.*

**[0:00-0:10]** "Hi, I'm [Name] and this is [Co-founder Name]. We're building StockMate — an AI copilot for stock trading."

**[0:10-0:25]** "Right now, retail traders are stuck piecing together 5-6 different tools — TradingView for charts, Twitter for sentiment, screeners for picks — and they're still guessing on when to buy, where to set stops, and how much to risk. It takes hours and most people get it wrong."

**[0:25-0:40]** "StockMate runs professional-grade technical analysis — the same 15 indicators hedge fund analysts use — and our AI synthesizes it into a specific trade plan: exact entry price, stop loss, profit targets, and position size. In seconds, not hours."

**[0:40-0:50]** "We have a production iOS app, a working backend with 34 analysis tools, and support for multiple AI models. We're ready to launch."

**[0:50-0:60]** "We're applying to YC because we've built the product — now we need help cracking distribution and navigating fintech. Thanks for watching."

---

*Sections marked with [FILL IN] require your personal information, metrics, and experiences. Fill these in with honest, specific details — YC values authenticity over polish.*
