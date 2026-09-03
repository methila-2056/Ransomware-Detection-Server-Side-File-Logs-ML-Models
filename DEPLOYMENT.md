# Ransomware Detection System - Deployment

## Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open browser
# Visit http://localhost:5000
```

## Deployment as a Service (Windows)

### Using a .bat file
Create `start_system.bat`:
```batch
@echo off
cd /d %~dp0
python app.py
pause
```

### Using NSSM (Non-Sucking Service Manager)
```batch
nssm install RansomwareDetection "C:\Python311\python.exe" "C:\path\to\app.py"
nssm start RansomwareDetection
```

## Production Considerations

- **Security**: Run behind a reverse proxy (Nginx/Caddy) with HTTPS
- **Monitoring**: Configure systemd/NSSM for auto-restart
- **Data**: Regular backups of `ransomware_detection.db`
- **Models**: Keep `models/` directory with write permissions
- **Ports**: Default port 5000, configurable in `app.py`

## Environment Variables

The system uses these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 5000 | Web server port |
| HOST | 0.0.0.0 | Bind address |

## Troubleshooting

### Models not loading
Ensure `models/` directory exists with all .joblib files. Delete models to force retraining.

### Port already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <pid> /F
```

### Watchdog errors
Ensure you have required permissions to monitor the target folders.
