from opentelemetry import trace
from prometheus_client import Counter, Histogram

# Metrics as specified in requirements
request_count = Counter('request_total', 'Total requests')
