# Deployment Guide

## Quick Start - GitHub Setup

### 1. Initialize Git Repository

```bash
cd MBA8583
git init
git add .
git commit -m "Initial commit: Philadelphia Educational Desert Explorer"
```

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Name: `philly-education-desert`
3. Description: "Interactive dashboard for analyzing educational access in Philadelphia"
4. Public or Private (your choice)
5. **Do NOT** initialize with README (we have one)

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/philly-education-desert.git
git branch -M main
git push -u origin main
```

## Streamlit Community Cloud Deployment

### Prerequisites
- GitHub repository (created above)
- Census API key (get free at https://api.census.gov/data/key_signup.html)

### Steps

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Repository: `YOUR_USERNAME/philly-education-desert`
   - Branch: `main`
   - Main file path: `app_block_groups.py`

3. **Add Secrets**
   - Click "Advanced settings"
   - In "Secrets" section, add:
   ```toml
   CENSUS_API_KEY = "your_actual_api_key_here"
   ```

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for build
   - Your app will be live at: `https://YOUR_USERNAME-philly-education-desert.streamlit.app`

### Post-Deployment

- App auto-updates when you push to GitHub
- Check logs if errors occur
- Manage secrets in Streamlit Cloud dashboard

## Alternative: Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app_block_groups.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build
docker build -t philly-education-desert .

# Run with API key
docker run -p 8501:8501 \
  -e CENSUS_API_KEY=your_key_here \
  philly-education-desert
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - CENSUS_API_KEY=${CENSUS_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

## Heroku Deployment

### Setup

1. **Install Heroku CLI**
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

2. **Login**
```bash
heroku login
```

3. **Create App**
```bash
heroku create philly-education-desert
```

### Configure Buildpacks

```bash
heroku buildpacks:add --index 1 https://github.com/heroku/heroku-geo-buildpack.git
heroku buildpacks:add --index 2 heroku/python
```

### Set Config Vars

```bash
heroku config:set CENSUS_API_KEY=your_key_here
```

### Create Procfile

Create `Procfile`:
```
web: sh setup.sh && streamlit run app_block_groups.py
```

### Create Setup Script

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### Deploy

```bash
git push heroku main
```

Access at: `https://philly-education-desert.herokuapp.com`

## AWS EC2 Deployment

### Launch Instance

1. **Create EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t2.medium (recommended)
   - Security group: Allow ports 22 (SSH) and 8501 (Streamlit)

2. **Connect via SSH**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Setup Application

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv gdal-bin libgdal-dev

# Clone repository
git clone https://github.com/YOUR_USERNAME/philly-education-desert.git
cd philly-education-desert

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Set environment variable
echo "export CENSUS_API_KEY='your_key_here'" >> ~/.bashrc
source ~/.bashrc
```

### Run with systemd

Create `/etc/systemd/system/philly-edu.service`:

```ini
[Unit]
Description=Philadelphia Educational Desert Explorer
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/philly-education-desert
Environment="CENSUS_API_KEY=your_key_here"
ExecStart=/home/ubuntu/philly-education-desert/venv/bin/streamlit run app_block_groups.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable philly-edu
sudo systemctl start philly-edu
```

### Setup Nginx Reverse Proxy

```bash
sudo apt install -y nginx

# Create nginx config
sudo nano /etc/nginx/sites-available/philly-edu
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/philly-edu /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Data Update Strategy

### Manual Updates

```bash
# SSH into server or run locally
python fetch_block_groups_live.py
python fetch_enhanced_k12_data.py

# Restart app (Streamlit Cloud auto-restarts, others need manual restart)
```

### Automated Updates (Cron)

```bash
crontab -e
```

Add (runs quarterly on 1st of Jan, Apr, Jul, Oct at 2 AM):
```cron
0 2 1 1,4,7,10 * cd /path/to/philly-education-desert && /path/to/venv/bin/python fetch_block_groups_live.py && /path/to/venv/bin/python fetch_enhanced_k12_data.py
```

## Monitoring & Maintenance

### Streamlit Cloud
- Built-in analytics at https://share.streamlit.io
- View logs in dashboard
- Auto-scaling included

### Self-Hosted

**Logging**:
```python
# Add to app_block_groups.py
import logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

**Monitoring**:
```bash
# Check service status
sudo systemctl status philly-edu

# View logs
journalctl -u philly-edu -f

# Resource usage
htop
```

## Security Best Practices

### API Key Management

1. **Never commit secrets to Git**
   ```bash
   # Check .gitignore includes:
   MyKeys/
   *.env
   .streamlit/secrets.toml
   ```

2. **Use environment variables**
   - Streamlit Secrets (cloud)
   - .env files (local)
   - System env vars (production)

3. **Rotate keys periodically**
   - Census API: Generate new key
   - Update in all environments

### Access Control

- **Streamlit Cloud**: Use "Share" settings to restrict access
- **Self-Hosted**: Use nginx basic auth or OAuth proxy

### HTTPS

**Streamlit Cloud**: Automatic

**Self-Hosted with Let's Encrypt**:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
pip install --upgrade -r requirements.txt
```

**GDAL errors on deployment**:
- Add gdal-bin to system packages
- Use pyogrio instead of fiona for better compatibility

**Out of memory**:
- Increase instance size (t2.medium minimum recommended)
- Clear Streamlit cache regularly

**Census API rate limits**:
- Free tier: 500 requests/day
- Cache data locally
- Use batch requests

### Getting Help

- GitHub Issues: https://github.com/YOUR_USERNAME/philly-education-desert/issues
- Streamlit Community: https://discuss.streamlit.io
- Stack Overflow: Tag with `streamlit`, `geopandas`

## Cost Estimates

| Platform | Cost | Notes |
|----------|------|-------|
| Streamlit Cloud | Free | Public apps, generous limits |
| Heroku | $7/month | Hobby tier |
| AWS EC2 | ~$15/month | t2.medium + storage |
| DigitalOcean | $12/month | 2GB droplet |
| Docker on VPS | $5-10/month | Basic VPS |

## Next Steps

1. ✅ Push to GitHub
2. ✅ Deploy to Streamlit Cloud
3. ✅ Add Census API key to secrets
4. ✅ Test live deployment
5. ✅ Set up automated data updates
6. ✅ Configure custom domain (optional)
7. ✅ Enable analytics/monitoring

## Support

For deployment issues, contact:
- Email: your.email@example.com
- GitHub: @YOUR_USERNAME
