I’m a solo founder building a lightweight, real-time AI evaluation prototype for a startup targeting fractional real estate (e.g., property price valuation) and money transfer fintechs (e.g., transaction fraud detection) in the UAE. I’m using Cursor AI to generate this code and need a Python-based solution that’s simple, modular, and ready for piloting. The prototype should include:

1. **Drift Detection**: Use KL divergence to compare real-time data distributions (e.g., property prices or transaction amounts) against a baseline dataset, flagging when drift exceeds a threshold (e.g., 0.1). Assume baseline data is a CSV file, and real-time data comes as a stream (simulated via a list or random generation for now).
2. **Anomaly Detection**: Implement outlier detection using Z-scores (e.g., flag property prices or transaction volumes >3 standard deviations from the mean) with real-time alerts.
3. **Lightweight Design**: Use minimal dependencies (e.g., NumPy, Pandas, SciPy) and run in a single script with clear outputs (e.g., print statements or logs) for easy integration into production pipelines.
4. **Fake Data Generation**: Include a simple function to generate synthetic baseline and real-time data (e.g., property prices between $10K-$1M, transaction amounts between $10-$10K) to demo the tool without real client data.
5. **Modularity**: Structure the code with functions/classes so I can swap in real data later (e.g., from Stake’s property listings or Ziina’s transactions).
6. **Out-of-the-Box Usability**: Include a main loop simulating real-time monitoring, with configurable settings (e.g., threshold, window size) and comments explaining each part for a non-expert like me.

Keep it optimized for a solo dev using Cursor AI to tweak later. Use Python 3.9+, and include  complex frameworks (e.g., TensorFlow) only after user confirmation

This prompt is expected to generate a script that:
Simulates a baseline dataset and a real-time stream as appropriate.
Computes KL divergence between sliding windows of real-time data and the baseline, flagging drift.
Detects outliers in the real-time stream using Z-scores, issuing alerts.
Logs results for pilot validation with a real-time data source. and api of a platform like stake or ziina.
Runs as a standalone script, ready to demo or integrate.