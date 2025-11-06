# Implementation Plan - HRL Finance UI

- [x] 1. Set up project structure and dependencies
  - Create backend directory with FastAPI project structure
  - Create frontend directory with React + TypeScript + Vite setup
  - Set up requirements.txt with FastAPI, uvicorn, python-socketio, pydantic dependencies
  - Set up package.json with React, TypeScript, Tailwind CSS, Recharts, Axios, Socket.IO client
  - Create .gitignore files for both backend and frontend
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 2. Implement backend API foundation
  - [x] 2.1 Create FastAPI application entry point
    - Implement main.py with FastAPI app initialization
    - Configure CORS middleware for frontend communication
    - Set up static file serving for frontend build
    - Add health check endpoint
    - _Requirements: 9.1, 9.5, 9.8_
  
  - [x] 2.2 Implement Pydantic models for requests and responses
    - ✅ Created EnvironmentConfig, TrainingConfig, RewardConfig, ScenarioConfig models
    - ✅ Created TrainingRequest, SimulationRequest, ReportRequest models
    - ✅ Created TrainingProgress, TrainingStatus, EpisodeResult, SimulationResults models
    - ✅ Created ScenarioSummary, ModelSummary, list response models
    - ✅ Created ReportResponse, HealthCheckResponse, ErrorResponse models
    - ✅ Added comprehensive validation rules with Field constraints
    - ✅ Implemented proper model configuration (protected_namespaces, populate_by_name)
    - _Requirements: 2.2, 2.3, 3.1, 4.1, 9.5_
  
  - [x] 2.3 Implement file management utilities ⭐ **COMPLETED**
    - ✅ Created file_manager.py with comprehensive file operations (568 lines)
    - ✅ Implemented YAML config read/write/delete/list functions with auto-extension handling
    - ✅ Implemented PyTorch model save/load/delete/list functions with metadata support
    - ✅ Implemented JSON results save/read/list functions with subdirectory support
    - ✅ Added filename sanitization to prevent path traversal attacks (regex-based)
    - ✅ Added path validation to ensure operations stay within allowed directories
    - ✅ Implemented ensure_directories() and get_file_size_mb() utility functions
    - ✅ Added comprehensive error handling with descriptive messages
    - ✅ Updated FILE_MANAGER_README.md with detailed documentation and examples
    - ✅ All functions include proper type hints and docstrings
    - ✅ Security features: sanitization, validation, safe YAML/JSON parsing
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 3. Implement Scenarios API ⭐ **COMPLETED**
  - [x] 3.1 Create scenarios service layer
    - ✅ Implemented scenario_service.py with comprehensive business logic (280 lines)
    - ✅ Added create_scenario function with validation and YAML storage
    - ✅ Added get_scenario function that loads, parses, and validates YAML
    - ✅ Added list_scenarios function that scans configs directory with metrics
    - ✅ Added update_scenario function with rename support
    - ✅ Added delete_scenario function
    - ✅ Added get_templates function with 5 preset profiles (conservative, balanced, aggressive, young_professional, young_couple)
    - ✅ Integrated with file_manager utilities for secure file operations
    - ✅ Proper error handling with descriptive messages
    - ✅ Exported via backend/services/__init__.py
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 10.1_
  
  - [x] 3.2 Create scenarios API endpoints
    - ✅ Implemented GET /api/scenarios endpoint to list all scenarios with summary info
    - ✅ Implemented GET /api/scenarios/{name} endpoint to get scenario details
    - ✅ Implemented POST /api/scenarios endpoint to create new scenario (201 Created)
    - ✅ Implemented PUT /api/scenarios/{name} endpoint to update scenario
    - ✅ Implemented DELETE /api/scenarios/{name} endpoint to delete scenario
    - ✅ Implemented GET /api/scenarios/templates endpoint for preset templates
    - ✅ Added comprehensive error handling with appropriate HTTP status codes (400, 404, 409, 500)
    - ✅ Created response models (ScenarioSummary, ScenarioDetail, ScenarioCreateResponse, ScenarioUpdateResponse, ScenarioDeleteResponse, TemplateResponse)
    - ✅ Full integration with ScenarioService layer
    - ✅ OpenAPI documentation auto-generated at /docs
    - ✅ Updated backend/README.md with API usage examples
    - _Requirements: 2.1, 2.2, 2.6, 2.7, 9.1, 9.2, 9.3, 9.7_

- [x] 4. Implement Training API and WebSocket ⭐ **COMPLETED**
  - [x] 4.1 Create training service layer ⭐ **COMPLETED**
    - ✅ Implemented training_service.py with HRL training orchestration (535 lines)
    - ✅ Added start_training function that initializes environment and agents
    - ✅ Implemented async training loop with progress tracking
    - ✅ Added stop_training function for graceful termination
    - ✅ Implemented model saving at intervals (checkpoints + final models)
    - ✅ Progress callback mechanism for WebSocket integration
    - ✅ Comprehensive error handling and status tracking
    - ✅ Integration with existing HRL components (BudgetEnv, Agents, HRLTrainer)
    - _Requirements: 3.1, 3.2, 3.6, 3.8, 10.2_
  
  - [x] 4.2 Implement WebSocket for real-time updates ⭐ **COMPLETED**
    - ✅ Created training_socket.py with Socket.IO server setup
    - ✅ Implemented WebSocket connection handler
    - ✅ Added emit_progress function to send TrainingProgress updates
    - ✅ Implemented training event broadcasting (started, progress, completed, stopped, error)
    - ✅ Added connection/disconnection handlers
    - ✅ TrainingSocketManager class for organized event management
    - ✅ Global socket_manager instance for easy access
    - _Requirements: 3.3, 3.4_
  
  - [x] 4.3 Create training API endpoints ⭐ **COMPLETED**
    - ✅ Implemented POST /api/training/start endpoint (202 Accepted)
    - ✅ Implemented POST /api/training/stop endpoint
    - ✅ Implemented GET /api/training/status endpoint (returns TrainingStatus)
    - ✅ Added background task management for training (asyncio)
    - ✅ Integrated WebSocket updates in training loop (every episode)
    - ✅ Comprehensive error handling with appropriate HTTP status codes
    - ✅ Integration with training_service and socket_manager
    - ✅ Updated main.py to mount Socket.IO with FastAPI
    - _Requirements: 3.1, 3.2, 3.6, 3.7, 3.8, 9.2_

- [x] 5. Implement Simulation API ⭐ **COMPLETED**
  - [x] 5.1 Create simulation service layer ⭐ **COMPLETED**
    - ✅ Implemented simulation_service.py with evaluation logic (392 lines)
    - ✅ Added run_simulation function that loads model and scenario
    - ✅ Implemented evaluation loop collecting episode data with deterministic policy
    - ✅ Added _calculate_statistics function for aggregating results
    - ✅ Implemented results saving to JSON files in results/simulations/
    - ✅ Added get_simulation_results function for retrieving saved results
    - ✅ Added list_simulations function for listing all past simulations
    - ✅ Integrated with existing HRL components (BudgetEnv, Agents, Analytics)
    - ✅ Model loading with proper agent initialization
    - ✅ Comprehensive error handling with descriptive messages
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 10.3_
  
  - [x] 5.2 Create simulation API endpoints ⭐ **COMPLETED**
    - ✅ Implemented POST /api/simulation/run endpoint (202 Accepted)
    - ✅ Implemented GET /api/simulation/results/{id} endpoint
    - ✅ Implemented GET /api/simulation/history endpoint with SimulationHistoryResponse
    - ✅ Added comprehensive error handling (404, 500)
    - ✅ Integration with simulation_service layer
    - ✅ Response models (SimulationResults, SimulationHistoryResponse)
    - ✅ Proper HTTP status codes and error responses
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 9.2, 9.3_

- [ ] 6. Implement Models API
  - Create GET /api/models endpoint to list trained models
  - Implement GET /api/models/{name} endpoint for model details
  - Implement DELETE /api/models/{name} endpoint
  - Add model metadata extraction from training history
  - _Requirements: 9.2, 9.3, 10.2, 10.5_

- [ ] 7. Implement Reports API
  - [ ] 7.1 Create report generation service
    - Implement report_service.py with PDF/HTML generation
    - Add generate_pdf_report function using ReportLab or WeasyPrint
    - Add generate_html_report function with templates
    - Implement report data aggregation from simulation results
    - Add chart image generation for reports
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [ ] 7.2 Create reports API endpoints
    - Implement POST /api/reports/generate endpoint
    - Implement GET /api/reports/{id} endpoint for download
    - Implement GET /api/reports/list endpoint
    - Add file serving for generated reports
    - _Requirements: 6.1, 6.2, 6.5, 6.6, 9.2_

- [ ] 8. Implement frontend foundation
  - [ ] 8.1 Set up React app structure
    - Create App.tsx with React Router setup
    - Implement routing for all pages
    - Create layout component with navigation
    - Set up Tailwind CSS configuration
    - Add theme provider for light/dark mode
    - _Requirements: 1.1, 8.1, 8.2, 8.6_
  
  - [ ] 8.2 Create API client service
    - Implement api.ts with Axios instance
    - Add functions for all API endpoints
    - Implement error handling and retry logic
    - Add request/response interceptors
    - _Requirements: 9.1, 9.5, 9.7_
  
  - [ ] 8.3 Create WebSocket client service
    - Implement websocket.ts with Socket.IO client
    - Add connection management functions
    - Implement event listeners for training updates
    - Add reconnection logic
    - _Requirements: 3.3, 3.4_

- [ ] 9. Implement Dashboard page
  - Create Dashboard.tsx component
  - Implement scenario cards grid with key metrics
  - Implement model cards grid with performance indicators
  - Add quick action buttons for creating scenarios and training
  - Implement recent activity feed
  - Add statistics summary cards
  - Fetch data from API on component mount
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [ ] 10. Implement Scenario Builder page
  - [ ] 10.1 Create ScenarioBuilder component
    - Implement form with all environment parameter inputs
    - Add form validation with error messages
    - Implement template selector dropdown
    - Create preview panel showing monthly cash flow
    - Add save and cancel buttons
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.8_
  
  - [ ] 10.2 Implement form state management
    - Use React hooks for form state
    - Implement input change handlers
    - Add validation logic for all fields
    - Implement template loading
    - Add API integration for saving scenarios
    - _Requirements: 2.2, 2.3, 2.4, 2.6, 2.7_

- [ ] 11. Implement Training Monitor page
  - [ ] 11.1 Create TrainingMonitor component
    - Implement training configuration form
    - Add start/pause/stop training buttons
    - Create progress bar component
    - Implement current metrics display cards
    - Add training status indicator
    - _Requirements: 3.1, 3.2, 3.6, 3.7, 3.8_
  
  - [ ] 11.2 Implement real-time chart updates
    - Create MetricChart component with Recharts
    - Implement WebSocket connection for training updates
    - Add chart data state management
    - Implement live chart updates as data arrives
    - Add multiple metric charts (reward, duration, stability)
    - _Requirements: 3.3, 3.4, 3.5_

- [ ] 12. Implement Simulation Runner page
  - Create SimulationRunner.tsx component
  - Implement model and scenario selector dropdowns
  - Add configuration form for number of episodes and seed
  - Implement run simulation button
  - Add progress indicator during simulation
  - Display summary statistics when complete
  - Add navigation to Results Viewer
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 13. Implement Results Viewer page
  - [ ] 13.1 Create ResultsViewer component
    - Implement summary statistics cards
    - Add tab navigation for different chart views
    - Create strategy learned display section
    - Add action buttons for compare, report, export
    - _Requirements: 5.1, 5.4, 5.5_
  
  - [ ] 13.2 Implement interactive charts
    - Create cash balance over time chart with Recharts
    - Create portfolio evolution chart
    - Create wealth accumulation chart
    - Create action distribution pie chart
    - Add tooltips on hover for all charts
    - Implement zoom and pan functionality
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8_

- [ ] 14. Implement Comparison view
  - Create Comparison.tsx component
  - Implement scenario selector for multiple scenarios
  - Create side-by-side metrics comparison table
  - Implement comparative charts
  - Add difference highlighting
  - Implement export to CSV/JSON functionality
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 15. Implement Report Generation
  - Create ReportModal component
  - Implement report configuration form
  - Add report type selector (PDF/HTML)
  - Implement section customization checkboxes
  - Add generate button with API integration
  - Implement download link when report is ready
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 16. Implement responsive design and accessibility
  - Add responsive breakpoints for all components
  - Implement mobile-friendly navigation menu
  - Add keyboard navigation support
  - Implement ARIA labels for all interactive elements
  - Test and fix color contrast issues
  - Add focus indicators for all focusable elements
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 17. Add error handling and loading states
  - Implement toast notification system
  - Add loading spinners for async operations
  - Create error boundary components
  - Add retry mechanisms for failed API calls
  - Implement graceful degradation for missing data
  - Add error logging
  - _Requirements: 9.7_

- [ ] 18. Create API documentation
  - Set up Swagger/OpenAPI documentation
  - Add endpoint descriptions and examples
  - Document request/response schemas
  - Add authentication documentation (if implemented)
  - Create API usage examples
  - _Requirements: 9.6_

- [ ] 19. Write tests
  - [ ] 19.1 Backend tests
    - Write unit tests for API endpoints
    - Write unit tests for service layer
    - Write integration tests for complete workflows
    - Write tests for WebSocket communication
    - _Requirements: All backend requirements_
  
  - [ ] 19.2 Frontend tests
    - Write component tests for all major components
    - Write tests for form validation
    - Write tests for API integration
    - Write E2E tests for critical user flows
    - _Requirements: All frontend requirements_

- [ ] 20. Create deployment configuration
  - Create Dockerfile for containerized deployment
  - Create docker-compose.yml for local development
  - Add environment variable configuration
  - Create production build scripts
  - Add deployment documentation
  - _Requirements: All requirements_

- [ ] 21. Create user documentation
  - Write README.md with setup instructions
  - Create user guide for all features
  - Add screenshots and GIFs for key workflows
  - Document API usage examples
  - Create troubleshooting guide
  - _Requirements: All requirements_
