# Deployment Guide

This document outlines deployment procedures for the health rankings analysis application.

## Deployment Platforms

### Streamlit Cloud (Recommended for Public Access)

Streamlit Cloud provides free hosting for public Streamlit applications.

**Prerequisites**:
- GitHub account
- Repository containing application files

**Deployment Steps**:

1. Prepare repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/repo.git
git push -u origin main
```

2. Configure Streamlit Cloud
- Navigate to share.streamlit.io
- Select "New app"
- Connect GitHub repository
- Configure settings:
  - Main file: `streamlit_app_modern.py`
  - Python version: 3.9+
- Deploy

**Environment Variables** (if needed):
- Add secrets in Streamlit Cloud dashboard
- Access via `st.secrets["KEY_NAME"]`

### Posit Connect (Enterprise Deployment)

For internal corporate deployments with access controls.

**Prerequisites**:
- Posit Connect server access
- API key credentials
- rsconnect-python package

**Installation**:
```bash
pip install rsconnect-python
```

**Deployment Command**:
```bash
rsconnect deploy streamlit \
  --server https://connect.company.com \
  --api-key YOUR_API_KEY \
  --title "Health Rankings Analysis" \
  streamlit_app_modern.py
```

**Git-Backed Deployment**:
1. Push code to internal Git repository
2. In Posit Connect UI: New Content → Import from Git
3. Select repository and branch
4. Configure automatic redeployment on push

### Local Development Server

For development and testing:

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run application
streamlit run streamlit_app_modern.py

# Access at http://localhost:8501
```

## Required Files

Ensure repository contains:
```
├── streamlit_app_modern.py
├── health_ml_analysis.py
├── us_health_2025.csv
├── requirements.txt
└── README.md
```

## Configuration

### requirements.txt
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
```

### .gitignore
```
venv/
__pycache__/
*.pyc
.DS_Store
.env
```

### Environment Variables

For sensitive configurations:

**Local development** (.env file):
```
API_KEY=your_key_here
DATABASE_URL=connection_string
```

**Production** (Streamlit Cloud secrets):
```toml
[secrets]
api_key = "your_key_here"
database_url = "connection_string"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["api_key"]
```

## Post-Deployment

### Verification Checklist
- [ ] Application loads without errors
- [ ] Data visualizations render correctly
- [ ] Interactive features function properly
- [ ] Page navigation works across all tabs
- [ ] Mobile responsiveness acceptable

### Monitoring
- Check application logs for errors
- Monitor usage metrics (if available)
- Set up alerts for downtime

### Updates
**Streamlit Cloud**: Push to connected branch
```bash
git add .
git commit -m "Update description"
git push origin main
```

**Posit Connect**: Redeploy or use git-backed auto-deploy

## Troubleshooting

### Common Issues

**Module Not Found**
- Verify all dependencies in requirements.txt
- Check Python version compatibility

**File Not Found**
- Ensure CSV file path is relative: `us_health_2025.csv`
- Verify file exists in repository

**Memory Errors**
- Consider data sampling for large datasets
- Optimize caching with @st.cache_data

**Performance Issues**
- Profile code for bottlenecks
- Implement lazy loading for heavy computations
- Cache expensive operations

### Platform-Specific Notes

**Streamlit Cloud**:
- Free tier: 1GB RAM limit
- Large files (>100MB): Use Git LFS or external storage
- Custom domains: Available on paid plans

**Posit Connect**:
- Resource limits: Configure in admin panel
- SSO integration: Configure with IT team
- Access control: Set via Connect dashboard

## Security Considerations

**Data Protection**:
- Do not commit sensitive data to public repositories
- Use environment variables for credentials
- Consider data anonymization for public deployments

**Access Control**:
- Streamlit Cloud: Public by default
- Posit Connect: Configure user/group permissions

## Support Resources

**Streamlit Documentation**: https://docs.streamlit.io  
**Posit Connect Guide**: https://docs.posit.co/connect  
**GitHub Actions**: For CI/CD automation

## Production Recommendations

1. Version control: Tag releases (`git tag v1.0.0`)
2. Testing: Validate locally before deployment
3. Documentation: Maintain changelog for updates
4. Monitoring: Track application health
5. Backups: Regular data backups if using mutable sources