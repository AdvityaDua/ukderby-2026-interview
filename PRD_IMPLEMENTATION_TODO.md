# PRD Implementation Todo Checklist

Implementation tracker for AI Interview Engine (No-Embedding Architecture).

## Progress
- Started: 2026-04-03
- Current status: In progress

## Checklist
- [ ] Lock runtime state contract
- [x] Create interview plan generator
- [x] Add company data cache layer
- [x] Define state persistence in Redis
- [x] Implement session load/save service
- [x] Wire run_turn orchestration API
- [x] Enforce one-turn concurrency guard
- [x] Implement token accounting pipeline
- [x] Add termination and routing tests
- [x] Add latency and token telemetry
- [x] Integrate coding-question UX flags
- [x] Run end-to-end interview validation

## Suggested Execution Order

### Phase 1: Contracts and Planning
- [ ] Lock runtime state contract
- [x] Create interview plan generator
- [x] Add company data cache layer

### Phase 2: State and Runtime Wiring
- [x] Define state persistence in Redis
- [x] Implement session load/save service
- [x] Wire run_turn orchestration API
- [x] Enforce one-turn concurrency guard

### Phase 3: Reliability and Observability
- [x] Implement token accounting pipeline
- [x] Add latency and token telemetry
- [x] Add termination and routing tests

### Phase 4: Product Integration
- [x] Integrate coding-question UX flags
- [x] Run end-to-end interview validation

## Notes
- Keep token counters cumulative per session: input_tokens, output_tokens, total_tokens.
- Ensure state writes are atomic around each run_turn call.
- Do not allow overlapping turn execution for the same session_id.
