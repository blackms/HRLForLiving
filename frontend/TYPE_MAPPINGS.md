# Frontend-Backend Type Mappings

This document shows how frontend TypeScript types map to backend Pydantic models.

## Configuration Types

### EnvironmentConfig
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Matches 1:1 with all fields:
- `income`, `fixed_expenses`, `variable_expense_mean`, `variable_expense_std`
- `inflation`, `safety_threshold`, `max_months`, `initial_cash`
- `risk_tolerance`, `investment_return_mean`, `investment_return_std`, `investment_return_type`

### TrainingConfig
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Frontend includes optional fields for display:
- `num_episodes`, `gamma_low`, `gamma_high`, `high_period`
- `batch_size`, `learning_rate_low`, `learning_rate_high`

### RewardConfig
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

All fields optional in frontend (uses backend defaults):
- `alpha`, `beta`, `gamma`, `delta`, `lambda_`, `mu`

## Request Types

### TrainingRequest
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Matches exactly:
- `scenario_name`, `num_episodes`, `save_interval`, `eval_episodes`, `seed?`

### SimulationRequest
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Matches exactly:
- `model_name`, `scenario_name`, `num_episodes`, `seed?`

### ReportRequest
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Matches exactly:
- `simulation_id`, `report_type`, `include_sections?`, `title?`

## Response Types

### TrainingProgress
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `episode`, `total_episodes`, `avg_reward`, `avg_duration`
- `avg_cash`, `avg_invested`, `stability`, `goal_adherence`, `elapsed_time`

### TrainingStatus
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `is_training`, `scenario_name?`, `current_episode?`, `total_episodes?`, `progress?`

### SimulationResult
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `simulation_id`, `scenario_name`, `model_name`, `num_episodes`, `timestamp`
- `duration_mean`, `duration_std`, `final_cash_mean`, `final_invested_mean`
- `final_portfolio_mean`, `total_wealth_mean`, `total_wealth_std`
- `investment_gains_mean`, `avg_invest_pct`, `avg_save_pct`, `avg_consume_pct`
- `episodes: EpisodeData[]`

### EpisodeData
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches backend `EpisodeResult`:
- `episode_num` (frontend) ↔️ `episode_id` (backend)
- `duration`, `final_cash`, `final_invested`, `final_portfolio_value`, `total_wealth`
- `investment_gains`, `avg_invest_pct`, `avg_save_pct`, `avg_consume_pct`
- `months?`, `cash?`, `invested?`, `portfolio_value?`, `actions?`

### ModelSummary
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `name`, `scenario_name`, `size_mb`, `trained_at`, `has_metadata`
- `episodes?`, `income?`, `risk_tolerance?`, `final_reward?`
- `avg_reward?`, `max_reward?`, `final_duration?`, `final_cash?`, `final_invested?`

### ModelDetail
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `name`, `scenario_name`, `high_agent_path`, `low_agent_path`
- `size_mb`, `trained_at`, `has_metadata`, `has_history`, `episodes?`
- `metadata?`, `environment_config?`, `training_config?`, `reward_config?`
- `training_history?`, `final_metrics?`

### Report
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches backend `ReportResponse`:
- `report_id`, `simulation_id`, `report_type`, `title`
- `generated_at`, `file_path`, `file_size_kb`, `sections`
- `status?`, `message?`

## Scenario Types

### Scenario
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/requests.py`

Matches backend `ScenarioConfig`:
- `name`, `description?`, `environment`, `training?`, `reward?`
- `created_at?`, `updated_at?`

### ScenarioSummary
**Frontend:** `src/types/index.ts` → **Backend:** `backend/models/responses.py`

Matches exactly:
- `name`, `description?`, `created_at?`, `updated_at?`
- `income`, `fixed_expenses`, `available_income_pct`, `risk_tolerance`

## Type Safety Benefits

1. **Compile-time validation**: TypeScript catches type mismatches before runtime
2. **IntelliSense support**: IDEs provide autocomplete and inline documentation
3. **Refactoring safety**: Changes to types are caught across the codebase
4. **API contract enforcement**: Frontend and backend stay in sync
5. **Reduced bugs**: Type errors caught during development, not production

## Maintaining Type Sync

When updating types:

1. **Backend changes**: Update Pydantic models in `backend/models/`
2. **Frontend changes**: Update TypeScript types in `frontend/src/types/index.ts`
3. **Documentation**: Update this file and relevant READMEs
4. **Testing**: Verify API integration tests pass

## Common Patterns

### Optional Fields
Backend uses Pydantic `Optional[T]` → Frontend uses `field?: T`

### Timestamps
Backend uses `datetime` → Frontend uses `string` (ISO 8601 format)

### Enums
Backend uses `Literal["a", "b"]` → Frontend uses `"a" | "b"`

### Nested Objects
Both use nested interfaces/models with matching structure

### Arrays
Backend uses `List[T]` → Frontend uses `T[]`

### Dictionaries
Backend uses `Dict[str, Any]` → Frontend uses `Record<string, any>`
