I’m a solo founder building a lightweight, real-time AI evaluation prototype for a startup targeting fractional real estate (e.g., Stake’s property price valuation) and money transfer fintechs (e.g., Ziina’s transaction fraud detection) in the UAE. I’m using Cursor AI to generate this Python-based solution, which should be simple, modular, and pilot-ready. My existing codebase (from a prior prompt) includes drift detection, anomaly detection, synthetic data generation, and a basic dashboard, but I need to enhance it based on new requirements. The updated prototype should include:

1. **Drift Detection**: Use KL divergence to compare real-time data distributions (e.g., property prices or transaction amounts) against a baseline dataset from a CSV file (e.g., 'baseline.csv'), flagging when drift exceeds a configurable threshold (default 0.1). Real-time data now comes from a mock API or external source (see below), with synthetic streams as a fallback.
2. **Anomaly Detection**: Implement outlier detection using Z-scores (e.g., flag property prices or transaction volumes >3 standard deviations from the mean), issuing real-time alerts tied to predefined problem statements (e.g., “Outlier Listing”).
3. **API Data Fetching**: Add a mock API call simulating a production system (e.g., Stake’s API returning JSON: {"price": 500000, "timestamp": "2025-02-23"}) to replace synthetic real-time data. Include a toggle (e.g., `USE_API = True/False`) to switch between API and synthetic data.
4. **External Data Source**: Add a mock external data fetch (e.g., simulating Property Finder listings as JSON: {"listing_price": 600000, "date": "2025-02-23"}) for comparison against API data. Support synthetic fallback for testing.
5. **Scheduled Runs**: Wrap the monitoring loop in a scheduler (using Python’s `schedule` library) to run every X seconds (default 60) or on new data availability, with an option for manual runs.
6. **Predefined Problem Statements**: Define a config list of deviation types (e.g., {"Price Drift": {"metric": "kl", "threshold": 0.1}, "Outlier Listing": {"metric": "zscore", "threshold": 3}}) linking drift/anomaly results to domain-specific issues. Output actionable reports (e.g., “Problem: Price Drift, Drift: 0.15, Action: Retrain model”).
7. **Lightweight Design**: Use minimal dependencies (e.g., NumPy, Pandas, SciPy, `schedule`, `json`) and keep it a single script with clear outputs (e.g., print statements and CSV logs) for easy integration into production pipelines.
8. **Fake Data Generation**: Retain a function to generate synthetic baseline (e.g., property prices $10K-$1M) and real-time data (e.g., prices with drift or outliers) as a fallback when `USE_API = False`.
9. **Modularity**: Structure with functions/classes (e.g., `fetch_data()`, `detect_drift()`, `generate_report()`) so I can swap in real API/external data later (e.g., from Stake or Ziina).
10. **Out-of-the-Box Usability**: Include a main loop simulating real-time monitoring, with configurable settings (e.g., drift threshold, window size, run frequency) and comments explaining each part for a non-expert like me.
11. **Report Generation**: Save results to a timestamped CSV (e.g., "report_2025-02-23.csv") with columns: timestamp, problem_type, metric, action_suggested.

Keep it optimized for a solo dev using Cursor AI to tweak later. Use Python 3.9+, and avoid complex frameworks (e.g., TensorFlow) unless I confirm. The script should:
- Simulate a baseline dataset (from CSV or synthetic) and real-time streams (API, external, or synthetic).
- Compute KL divergence between sliding windows of real-time data and the baseline, flagging drift.
- Detect outliers in the real-time stream using Z-scores, issuing alerts.
- Log results to CSV for pilot validation, bridging production data (API) and real-time sources (external).
- Run as a standalone script (<250 lines), ready to demo or integrate with Stake/Ziina.
