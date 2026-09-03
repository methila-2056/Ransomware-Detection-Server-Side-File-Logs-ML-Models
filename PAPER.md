# Research Paper Reference

This project is based on the following research paper:

## Citation

**Aranyi, G., Miseta, T., & Szucs, V. (2026).**
*Ransomware detection based on server-side file operation logs using machine learning.*
Journal on Information Security, 2026:8.

## Methodology Implemented

The system follows the paper's methodology for:

1. **Server-side file operation logging**
   - Captures 5 operation types: create, write, read, rename, delete
   - Aggregates operations into 1-second windows
   - Uses nanosecond precision timestamps

2. **Feature extraction**
   - 5 features per window: [nc, nw, nr, nm, nu]
   - Captures ransomware's characteristic file operation patterns

3. **Machine learning detection**
   - 5 classifiers trained and compared
   - XGBoost used as primary detector
   - GridSearchCV hyperparameter optimization
   - Recall (sensitivity) prioritized for attack detection

4. **Simulation parameters**
   - Realistic SME user profiles
   - 5 ransomware family attack patterns
   - Attack ramp-up and wind-down effects

## Implementation Differences

- **Web dashboard**: Real-time visualization via WebSocket
- **Real monitoring**: Extends paper's concept with actual folder monitoring
- **SQLite persistence**: Stores operation logs for training data accumulation
- **Multiple modes**: Simulation (with labels) and real monitoring (without labels)
