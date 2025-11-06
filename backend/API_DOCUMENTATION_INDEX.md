# HRL Finance System API - Documentation Index

Welcome to the HRL Finance System API documentation. This index will help you find the information you need.

## 📚 Documentation Structure

### 1. Quick Start Guide
**File:** [API_QUICK_START.md](./API_QUICK_START.md)

Perfect for getting started in 5 minutes. Includes:
- Installation and setup
- Complete workflow example
- Python and JavaScript client examples
- WebSocket real-time updates
- Common issues and tips

**Start here if you're new to the API!**

### 2. Complete API Documentation
**File:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

Comprehensive reference guide covering:
- All API endpoints with detailed descriptions
- Request/response schemas
- Error handling
- WebSocket events
- Best practices
- Complete workflow examples

**Use this as your primary reference.**

### 3. Backend README
**File:** [README.md](./README.md)

Backend-specific information including:
- Project structure
- Setup and configuration
- API endpoint overview
- Data models
- Usage examples
- Implementation status

**Read this for backend development.**

### 4. Interactive Documentation ✅ **ENHANCED**
**URLs (when server is running):**
- **API Root:** http://localhost:8000/ - Overview with navigation links and endpoint listing
- **Health Check:** http://localhost:8000/health - Service health status with timestamp
- **Swagger UI:** http://localhost:8000/docs - Interactive API testing with enhanced metadata
- **ReDoc:** http://localhost:8000/redoc - Alternative documentation view with better readability
- **OpenAPI Schema:** http://localhost:8000/openapi.json - Machine-readable API specification

**Enhanced Features:**
- Comprehensive API description with markdown formatting
- Organized endpoint tags (scenarios, training, simulation, models, reports)
- Getting started workflow guide (4 steps)
- WebSocket connection documentation
- Authentication and rate limiting notes
- Contact information and MIT license details

**Use these for interactive API exploration and testing.**

## 🎯 Quick Navigation

### By Use Case

#### I want to...

**Get started quickly**
→ [API_QUICK_START.md](./API_QUICK_START.md)

**Learn about a specific endpoint**
→ [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) → API Endpoints section

**Understand request/response formats**
→ [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) → Request/Response Examples
→ [README.md](./README.md) → Data Models section

**Implement WebSocket updates**
→ [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) → WebSocket Events
→ [API_QUICK_START.md](./API_QUICK_START.md) → WebSocket Examples

**Handle errors properly**
→ [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) → Error Handling

**Follow best practices**
→ [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) → Best Practices

**Set up the backend**
→ [README.md](./README.md) → Setup section

**Understand the architecture**
→ [README.md](./README.md) → Project Structure

### By API Category

#### Scenarios API
Manage financial scenarios with customizable parameters.

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#scenarios)
- [README.md](./README.md#scenarios-api-usage)

**Endpoints:**
- `GET /api/scenarios` - List all scenarios
- `GET /api/scenarios/{name}` - Get scenario details
- `POST /api/scenarios` - Create new scenario
- `PUT /api/scenarios/{name}` - Update scenario
- `DELETE /api/scenarios/{name}` - Delete scenario
- `GET /api/scenarios/templates` - Get preset templates

#### Training API
Train AI models using hierarchical reinforcement learning.

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#training)
- [README.md](./README.md#training-api-usage)

**Endpoints:**
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `GET /api/training/status` - Get training status

**WebSocket:** `/socket.io` for real-time updates

#### Simulation API
Run simulations with trained models.

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#simulation)
- [README.md](./README.md#simulation-api-usage)

**Endpoints:**
- `POST /api/simulation/run` - Run simulation
- `GET /api/simulation/results/{id}` - Get simulation results
- `GET /api/simulation/history` - List past simulations

#### Models API
Manage trained models.

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#models)
- [README.md](./README.md) → Models API section

**Endpoints:**
- `GET /api/models` - List all models
- `GET /api/models/{name}` - Get model details
- `DELETE /api/models/{name}` - Delete model

#### Reports API
Generate and download comprehensive reports.

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#reports)
- [README.md](./README.md#reports-api-usage)

**Endpoints:**
- `POST /api/reports/generate` - Generate report
- `GET /api/reports/{id}` - Download report
- `GET /api/reports/list` - List generated reports
- `GET /api/reports/{id}/metadata` - Get report metadata

## 🔍 Finding Specific Information

### Request/Response Schemas

**Pydantic Models:**
- Request models: `backend/models/requests.py`
- Response models: `backend/models/responses.py`

**Documentation:**
- [README.md](./README.md#data-models) - Detailed model descriptions
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) - Examples for each endpoint
- Interactive docs: http://localhost:8000/docs - Auto-generated schemas

### Code Examples

**Python:**
- [API_QUICK_START.md](./API_QUICK_START.md#-python-client-example)
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#complete-workflow-example)

**JavaScript/TypeScript:**
- [API_QUICK_START.md](./API_QUICK_START.md#-javascripttypescript-client-example)

**cURL:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) - Every endpoint includes cURL examples
- [README.md](./README.md) - Usage examples for each API

**WebSocket:**
- [API_QUICK_START.md](./API_QUICK_START.md#-websocket-real-time-updates)
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#websocket-events)

### Error Handling

**Documentation:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#error-handling) - Complete error reference
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#common-error-scenarios) - Solutions

**HTTP Status Codes:**
- `200 OK` - Success
- `201 Created` - Resource created
- `202 Accepted` - Request accepted for processing
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource already exists
- `500 Internal Server Error` - Server error

### Configuration

**Environment Variables:**
- [README.md](./README.md#-configuration)

**CORS Setup:**
- [README.md](./README.md#cors-configuration)

**Server Configuration:**
- [README.md](./README.md#setup)

## 📖 Learning Path

### Beginner

1. **Start here:** [API_QUICK_START.md](./API_QUICK_START.md)
2. **Try the examples:** Run the Python or JavaScript examples
3. **Explore interactively:** http://localhost:8000/docs
4. **Read the overview:** [README.md](./README.md)

### Intermediate

1. **Deep dive:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
2. **Study data models:** [README.md](./README.md#data-models)
3. **Implement WebSocket:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#websocket-events)
4. **Handle errors:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#error-handling)

### Advanced

1. **Best practices:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#best-practices)
2. **Security:** [README.md](./README.md#-security-considerations)
3. **Performance:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#best-practices)
4. **Production deployment:** [README.md](./README.md#production-deployment)

## 🛠️ Development Resources

### API Testing

**Interactive Tools:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Command Line:**
```bash
# Health check
curl http://localhost:8000/health

# List scenarios
curl http://localhost:8000/api/scenarios

# Get OpenAPI schema
curl http://localhost:8000/openapi.json
```

**Python:**
```python
import requests
response = requests.get('http://localhost:8000/health')
print(response.json())
```

### Code Structure

**Backend Structure:**
```
backend/
├── main.py                 # FastAPI app entry point
├── api/                    # API endpoints
├── services/               # Business logic
├── models/                 # Pydantic models
├── websocket/              # WebSocket handlers
└── utils/                  # Utilities
```

**Documentation:**
- [README.md](./README.md#-project-structure)

### Testing

**Documentation:**
- [README.md](./README.md#-testing)

**Run Tests:**
```bash
pytest
pytest --cov=backend
```

## 🆘 Getting Help

### Troubleshooting

**Common Issues:**
- [API_QUICK_START.md](./API_QUICK_START.md#-common-issues)
- [README.md](./README.md#-troubleshooting)

**Error Reference:**
- [API_DOCUMENTATION.md](./API_DOCUMENTATION.md#error-handling)

### Support Channels

- **GitHub Issues:** Report bugs and request features
- **GitHub Discussions:** Ask questions and share ideas
- **Documentation:** This index and linked documents

## 📝 Contributing

### Documentation

When adding new features:
1. Update [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) with endpoint details
2. Add examples to [API_QUICK_START.md](./API_QUICK_START.md) if applicable
3. Update [README.md](./README.md) with implementation status
4. Ensure OpenAPI docs are auto-generated correctly

### Code

1. Add docstrings to all endpoints
2. Include request/response examples in docstrings
3. Update Pydantic models with descriptions
4. Add tests for new endpoints

## 🔄 Version History

### Version 1.0.0 (Current)

**Features:**
- Complete Scenarios API (CRUD operations)
- Training API with WebSocket support
- Simulation API with evaluation
- Models API for model management
- Reports API (HTML/PDF generation)
- Real-time training updates
- Comprehensive error handling
- Full OpenAPI documentation

**Documentation:**
- Complete API documentation
- Quick start guide
- Interactive Swagger UI
- Code examples (Python, JavaScript)
- WebSocket examples

## 📚 External Resources

### Technologies Used

- **FastAPI:** https://fastapi.tiangolo.com/
- **Pydantic:** https://docs.pydantic.dev/
- **Socket.IO:** https://socket.io/docs/
- **Uvicorn:** https://www.uvicorn.org/

### Related Documentation

- **Frontend Documentation:** `../frontend/README.md`
- **Project README:** `../README.md`
- **Technical Paper:** `../TECHNICAL_PAPER.md`
- **Backend Tests:** `tests/README.md`
- **Integration Tests:** `tests/INTEGRATION_TESTS_SUMMARY.md`

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated:** 2024-01-15

**API Version:** 1.0.0

**Documentation Version:** 1.0.0
