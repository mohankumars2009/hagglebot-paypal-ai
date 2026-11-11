# 🤝 HaggleBot: Autonomous AI Deal-Maker & Dynamic PayPal Checkout

> Built for the **PayPal AI Hackathon 2026** (Targeting *Best Use of Agentic Commerce* & *Best Use of PayPal + AI*).

## 💡 Inspiration & Problem
E-commerce suffers from nearly 70% cart abandonment, largely driven by price friction. Static checkout pages cannot negotiate with price-sensitive buyers, students, or bulk shoppers.

## 🚀 What It Does
**HaggleBot** turns static product listings into interactive, margin-aware negotiations:
1. **Merchant Guardrails:** Merchants configure a public `list_price` and a confidential `floor_price`.
2. **Agentic Negotiation:** Buyers negotiate in natural language with an autonomous AI sales agent powered by **Google Gemini 2.5 Flash**.
3. **Instant Dynamic PayPal Order Minting:** The moment a mutually agreeable price is reached, HaggleBot invokes the **PayPal Orders V2 API** (`/v2/checkout/orders`) in real time to mint a custom-priced checkout link inside the chat and verifies payment capture status via `/v2/checkout/orders/{id}`.

## 🛠️ How to Build & Run Locally (For Judges)

### Prerequisites
* Python 3.10+
* A free [PayPal Developer Sandbox](https://developer.paypal.com/) Client ID & Secret
* A free [Google Gemini API Key](https://aistudio.google.com/)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/hagglebot-paypal-ai.git](https://github.com/YOUR_GITHUB_USERNAME/hagglebot-paypal-ai.git)
   cd hagglebot-paypal-ai
