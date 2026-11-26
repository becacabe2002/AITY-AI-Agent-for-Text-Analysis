# Domain models

The domain models are plain, immutable data shapes shared by services, repositories and the API.

- **User**: an account. The password hash stays inside the service layer.
- **Session**: a named workspace that groups one user's documents. It holds no model choice; provider and model are picked per analysis.
- **Document**: an uploaded file kept as text only. See [document upload](document-upload.md).
- **AnalysisResult**: the stored record of one analysis. See [analysis results](analysis-results.md).
- **SustainabilityMetrics**: the energy and CO2e estimate stored with each analysis. See [sustainability estimation](sustainability-estimation.md).
- **UserApiKey**: one encrypted cloud-provider key per user. See [API keys](api-keys.md).

Shared conventions:

- All timestamps are timezone-aware and stored in UTC; naive datetimes are rejected.
- Models validate themselves, so an inconsistent record cannot be created.
- API request and response shapes are separate from these models. See [API models](api-models.md).
