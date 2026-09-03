# API Reference

This document describes the REST and WebSocket API endpoints.

## REST API

### GET /api/status
Returns current system status including:
- Active mode
- Simulation/monitoring state
- Model training status
- Runtime metrics
- Simulator state
- Monitor state
- Model comparison data

### GET /api/history
Returns tick history and recent alerts for dashboard charts.

### GET /api/models
Returns model comparison data (accuracy, sensitivity, F1, etc.).

### GET /api/folders
Returns available monitoring folders with existence and selection status.

### GET /api/monitor_events
Returns recent file events from real monitoring.

### GET /api/stats
Returns database statistics including:
- System stats
- Recent alerts
- Attack sessions

## WebSocket Events

### Client → Server

| Event | Payload | Description |
|-------|---------|-------------|
| `start_simulation` | `{}` | Start simulation mode |
| `stop_simulation` | `{}` | Stop simulation mode |
| `start_monitoring` | `{folders: []}` | Start real folder monitoring |
| `stop_monitoring` | `{}` | Stop real monitoring |
| `force_attack` | `{family: "optional"}` | Force a simulated attack |
| `set_speed` | `{speed: 1.0}` | Set simulation speed |
| `reset_metrics` | `{}` | Reset all runtime metrics |

### Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `initial_data` | Full state | Sent on client connection |
| `tick_update` | Tick + metrics | Real-time 1-second updates |
| `mode_status` | Status | Mode start/stop confirmation |
| `attack_forced` | Success | Forced attack confirmation |
| `speed_updated` | Speed | Speed change confirmation |
| `metrics_reset` | Status | Metrics reset confirmation |

## Tick Data Structure

```json
{
  "timestamp": 1234567890,
  "timestamp_ns": 1234567890123456789,
  "nc": 45,
  "nw": 60,
  "nr": 120,
  "nm": 15,
  "nu": 8,
  "att": 0,
  "user": "Secretary",
  "family": null,
  "predictions": {
    "rf": {"name": "rf", "prediction": 0, "probability": 0.12, "confidence": 98.2},
    "xgb": {"name": "xgb", "prediction": 0, "probability": 0.08, "confidence": 99.1}
  },
  "file_events": []
}
```
