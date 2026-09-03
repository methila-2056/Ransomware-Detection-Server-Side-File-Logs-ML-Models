# Ransomware Detection System - Roadmap

Planned improvements and future developments for the Ransomware Detection System.

## Near-Term (v1.1)

- [ ] Add authentication for dashboard access
- [ ] Implement email/SMS alert notifications
- [ ] Add CSV export for detection logs
- [ ] Improve real monitor file event detail

## Medium-Term (v1.2)

- [ ] Add anomaly detection with unsupervised learning (Isolation Forest, One-Class SVM)
- [ ] Support for network-based file operation monitoring
- [ ] Docker container deployment
- [ ] Multi-user role-based access control

## Long-Term (v2.0)

- [ ] Deep learning models (LSTM for temporal patterns)
- [ ] Real-time model retraining pipeline
- [ ] Centralized monitoring across multiple machines
- [ ] Integration with SIEM platforms (Splunk, ELK)
- [ ] Threat intelligence feed integration
- [ ] Automated incident response actions

## Research Extensions

- [ ] Larger ransomware family dataset
- [ ] Cross-organization validation study
- [ ] Adversarial robustness testing
- [ ] Model interpretability analysis (SHAP values)

## Known Limitations

- Real monitoring lacks ground-truth labels
- Simulation patterns based on SME workloads only
- Windows-focused (uses ReadDirectoryChangesW)
- No persistence of trained models to version control

## Acknowledgments

- Aranyi, G., Miseta, T., & Szucs, V. (2026) for the foundational research
- Scikit-learn, XGBoost, Flask, and Watchdog open source communities
