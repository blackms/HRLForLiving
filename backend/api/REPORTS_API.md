# Reports API Documentation

The Reports API provides endpoints for generating, downloading, and managing PDF and HTML reports from simulation results.

## Endpoints

### 1. Generate Report

Generate a comprehensive report from simulation results.

**Endpoint:** `POST /api/reports/generate`

**Request Body:**
```json
{
  "simulation_id": "benedetta_case_bologna_coppia_1730000000",
  "report_type": "html",
  "include_sections": ["summary", "scenario", "results", "strategy"],
  "title": "Financial Analysis Report - Benedetta Case"
}
```

**Parameters:**
- `simulation_id` (required): ID of the simulation results to generate report from
- `report_type` (required): Format of the report - either "pdf" or "html"
- `include_sections` (optional): List of sections to include. Available sections:
  - `summary`: Summary statistics
  - `scenario`: Scenario configuration details
  - `training`: Training configuration
  - `results`: Detailed results breakdown
  - `strategy`: Strategy learned visualization
  - `charts`: Episode data and visualizations
- `title` (optional): Custom title for the report

**Response (202 Accepted):**
```json
{
  "report_id": "report_benedetta_case_bologna_coppia_1730000000_1730000100",
  "simulation_id": "benedetta_case_bologna_coppia_1730000000",
  "report_type": "html",
  "title": "Financial Analysis Report - Benedetta Case",
  "generated_at": "2024-11-06T10:30:00",
  "file_path": "reports/report_benedetta_case_bologna_coppia_1730000000_1730000100.html",
  "file_size_kb": 125.5,
  "sections": ["summary", "scenario", "results", "strategy"],
  "status": "completed",
  "message": "Report generated successfully: report_benedetta_case_bologna_coppia_1730000000_1730000100"
}
```

**Error Responses:**
- `404 Not Found`: Simulation results not found
- `400 Bad Request`: Invalid report type or parameters
- `500 Internal Server Error`: Report generation failed

**Example:**
```bash
curl -X POST "http://localhost:8000/api/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "benedetta_case_bologna_coppia_1730000000",
    "report_type": "html",
    "title": "My Financial Report"
  }'
```

---

### 2. Download Report

Download a generated report file.

**Endpoint:** `GET /api/reports/{report_id}`

**Parameters:**
- `report_id` (path): Unique report identifier

**Response:**
- File download (HTML or PDF)
- Content-Type: `text/html` or `application/pdf`
- Content-Disposition: `attachment; filename="report_id.html"`

**Error Responses:**
- `404 Not Found`: Report not found
- `500 Internal Server Error`: Error retrieving report

**Example:**
```bash
curl -X GET "http://localhost:8000/api/reports/report_benedetta_case_bologna_coppia_1730000000_1730000100" \
  -o report.html
```

---

### 3. List Reports

Get a list of all generated reports.

**Endpoint:** `GET /api/reports/list`

**Response (200 OK):**
```json
{
  "reports": [
    {
      "report_id": "report_benedetta_case_bologna_coppia_1730000000_1730000100",
      "simulation_id": "benedetta_case_bologna_coppia_1730000000",
      "report_type": "html",
      "title": "Financial Analysis Report - Benedetta Case",
      "generated_at": "2024-11-06T10:30:00",
      "file_path": "reports/report_benedetta_case_bologna_coppia_1730000000_1730000100.html",
      "file_size_kb": 125.5,
      "sections": ["summary", "scenario", "results", "strategy"]
    }
  ],
  "total": 1
}
```

**Error Responses:**
- `500 Internal Server Error`: Error listing reports

**Example:**
```bash
curl -X GET "http://localhost:8000/api/reports/list"
```

---

### 4. Get Report Metadata

Get metadata for a specific report without downloading the file.

**Endpoint:** `GET /api/reports/{report_id}/metadata`

**Parameters:**
- `report_id` (path): Unique report identifier

**Response (200 OK):**
```json
{
  "report_id": "report_benedetta_case_bologna_coppia_1730000000_1730000100",
  "simulation_id": "benedetta_case_bologna_coppia_1730000000",
  "report_type": "html",
  "title": "Financial Analysis Report - Benedetta Case",
  "generated_at": "2024-11-06T10:30:00",
  "file_path": "reports/report_benedetta_case_bologna_coppia_1730000000_1730000100.html",
  "file_size_kb": 125.5,
  "sections": ["summary", "scenario", "results", "strategy"]
}
```

**Error Responses:**
- `404 Not Found`: Report not found
- `500 Internal Server Error`: Error retrieving metadata

**Example:**
```bash
curl -X GET "http://localhost:8000/api/reports/report_benedetta_case_bologna_coppia_1730000000_1730000100/metadata"
```

---

## Report Sections

### Summary
Displays key statistics including:
- Average duration
- Total wealth (mean and std dev)
- Final cash balance
- Investment gains

### Scenario
Shows the scenario configuration:
- Financial parameters (income, expenses, inflation)
- Investment parameters (returns, volatility)
- Risk tolerance and safety thresholds

### Training
Displays training configuration:
- Number of episodes
- High-level planning period
- Discount factors

### Results
Detailed breakdown of results:
- Wealth components (cash, invested, portfolio)
- Statistical measures

### Strategy
Visualizes the learned strategy:
- Investment allocation percentage
- Savings allocation percentage
- Consumption allocation percentage

### Charts
Episode-level data visualization:
- Episode summaries table
- Cash balance over time
- Portfolio evolution
- Wealth accumulation

---

## Report Types

### HTML Reports
- Viewable in any web browser
- Responsive design for different screen sizes
- Print-friendly styling
- Smaller file size
- No additional dependencies required

### PDF Reports
- Professional document format
- Suitable for sharing and archiving
- Consistent rendering across platforms
- Requires WeasyPrint library
- Larger file size

**Note:** PDF generation requires the WeasyPrint library. If not installed, the API will generate an HTML report instead and return an error message with installation instructions.

To install WeasyPrint:
```bash
pip install weasyprint
```

---

## Usage Examples

### Python Example

```python
import requests

# Generate HTML report
response = requests.post(
    "http://localhost:8000/api/reports/generate",
    json={
        "simulation_id": "my_simulation_123",
        "report_type": "html",
        "include_sections": ["summary", "results", "strategy"],
        "title": "Q4 Financial Analysis"
    }
)

report_data = response.json()
report_id = report_data["report_id"]

# Download the report
report_file = requests.get(
    f"http://localhost:8000/api/reports/{report_id}"
)

with open("my_report.html", "wb") as f:
    f.write(report_file.content)
```

### JavaScript Example

```javascript
// Generate report
const response = await fetch('http://localhost:8000/api/reports/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    simulation_id: 'my_simulation_123',
    report_type: 'html',
    title: 'Q4 Financial Analysis'
  })
});

const reportData = await response.json();
const reportId = reportData.report_id;

// Download report
window.location.href = `http://localhost:8000/api/reports/${reportId}`;
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "timestamp": "2024-11-06T10:30:00"
}
```

Common error types:
- `NotFound`: Resource not found (404)
- `ValidationError`: Invalid request parameters (400)
- `DependencyError`: Missing required dependency (500)
- `ReportGenerationError`: Error during report generation (500)
- `RetrievalError`: Error retrieving report (500)
- `ListError`: Error listing reports (500)

---

## File Storage

Reports are stored in the `reports/` directory at the project root:
- Report files: `reports/{report_id}.html` or `reports/{report_id}.pdf`
- Metadata files: `reports/{report_id}_metadata.json`

Reports are persisted on disk and can be accessed later using the report ID.
