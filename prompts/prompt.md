Generate a modular, microarchitecture-based software application for AI-driven data management with the following requirements:

Architecture: Design a microservices-inspired system with loosely coupled, domain-driven modules (e.g., data ingestion, data processing, AI inference, data storage, and API gateway). Use hexagonal architecture (ports and adapters) within modules to decouple business logic from infrastructure. Enable communication via REST APIs (with OpenAPI specs) or an event-driven approach (e.g., Kafka, RabbitMQ) with retry mechanisms and idempotency for resilience.
Test-Driven Development (TDD): Adopt a TDD-first approach. Write unit tests (e.g., Jest, PyTest, JUnit) for core logic with 80%+ coverage, followed by integration tests for module interactions and contract tests for APIs. Use test doubles (mocks, stubs) to isolate dependencies and property-based testing (e.g., Hypothesis in Python) to catch edge cases.
Modularity and Maintainability: Build self-contained modules with single responsibilities, using a dependency injection framework (e.g., InversifyJS, FastAPI's Depends, Spring DI) or a plugin system. Enforce coding standards with linters (e.g., ESLint, Flake8) and provide a clear module contract (e.g., interface files, type definitions). Include a 'feature toggle' system to enable/disable experimental features without redeployment.
AI Data Management: Integrate an AI-driven capability (e.g., anomaly detection, clustering, or predictive modeling) using a lightweight ML framework (e.g., scikit-learn, TensorFlow Lite). Support pluggable models with a model registry (e.g., MLflow or a simple file-based system) and handle batch/streaming data (e.g., via Apache Kafka or file uploads). Expose results via a versioned API with caching (e.g., Redis) for performance.
Automation: Provide a fully automated setup with Docker containers for each module, orchestrated via Docker Compose (local) or Kubernetes manifests (production). Include a CI/CD pipeline (e.g., GitHub Actions, GitLab CI) with stages for linting, testing, building, and deployment. Add a Makefile or shell script for local dev tasks (e.g., make test, make run).
Sustainability: Optimize for resource efficiency (e.g., async processing with asyncio/Node.js event loop, minimal memory footprint with streaming data). Implement graceful degradation (e.g., fallback to simpler logic if AI fails) and ensure maintainability with auto-generated API docs (e.g., Swagger), structured logging (e.g., Winston, Log4j with correlation IDs), and a changelog.
Tech Stack: Default to Python with FastAPI for the backend, PostgreSQL for persistent storage, Redis for caching, and a simple CLI frontend (optional: React). Allow flexibility—confirm the stack and domain (e.g., financial data, IoT) before proceeding.
Resilience and Security: Add circuit breakers (e.g., Hystrix, resilience4j) for external calls, rate limiting on APIs, and basic security (e.g., JWT authentication, input validation with schemas like Pydantic/Zod). Include health-check endpoints (/health) per module.
Observability: Embed tracing (e.g., OpenTelemetry) and metrics (e.g., Prometheus) for monitoring. Log key events in a structured format and provide a dashboard spec (e.g., Grafana) for visualizing system health.
Deliverables: Provide:
A project structure with clear README (setup, architecture diagram, maintenance tips).
Sample code for two modules (e.g., ingestion + AI inference) with tests.
Automation scripts (Docker, CI/CD config).
A scalability plan (e.g., how to shard data or add nodes).
Constraints: Start with a small, functional scope (e.g., 1,000 records, 2 modules) but design for scale (e.g., horizontal scaling, sharding). Avoid premature optimization; prioritize simplicity, extensibility, and developer experience (DX).
Evolutionary Design: Enable iterative growth by using a strangler pattern—start with a monolith-like core that can be split into microservices later. Include a migration guide for evolving the system (e.g., from SQLite to PostgreSQL, or REST to gRPC).
Domain-Driven Design (DDD): Apply DDD principles—define a ubiquitous language for the data management domain (e.g., 'record', 'prediction') and align modules with bounded contexts. Provide a glossary in the README.
Ethical AI: Ensure the AI component logs its decision-making process for auditability (e.g., feature weights, confidence scores). Add a config option to disable AI if bias or ethical concerns arise.
Lateral Inspiration: Borrow from chaos engineering—include a script to simulate failures (e.g., network delays, DB outages) and test resilience. Draw from serverless paradigms by making modules stateless where possible, easing deployment.
Future-Proofing: Support polyglot persistence (e.g., swap PostgreSQL for MongoDB with minimal changes) and provide an extension point for new data types (e.g., images, time-series) via abstract data handlers.
Developer Experience (DX): Add a CLI tool for common tasks (e.g., app generate-module <name>), inline code comments, and a quickstart guide. Use static typing (e.g., TypeScript, Python's type hints) for better IDE support.
Customization Hook: Before generating, prompt the user: 'Please specify tech stack (default: Python/FastAPI, PostgreSQL), data domain (e.g., IoT, finance), and priority (e.g., speed, security). Otherwise, proceed with defaults.'

Additional Requirements:

Frontend Development:
- Implement a React-based frontend with TypeScript
- Use Material UI for consistent component design
- Create reusable components for data visualization and model monitoring

Enhanced Schema Validation:
- Implement comprehensive Pydantic models for all data structures
- Include example configurations in schema definitions
- Add validation rules with custom validators

Extended Docker Configuration:
- Multi-stage builds for optimized images
- Service dependencies and health checks
- Volume mounts for persistent data