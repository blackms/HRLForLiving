# Nginx Reverse Proxy Configuration

This directory contains Nginx configuration for production deployment with SSL, load balancing, and caching.

## Features

- **SSL/TLS Termination**: HTTPS support with automatic HTTP to HTTPS redirect
- **Load Balancing**: Distribute traffic across multiple backend instances
- **WebSocket Support**: Proper WebSocket proxying for real-time training updates
- **Rate Limiting**: Protect API endpoints from abuse
- **Caching**: Cache static assets and API responses
- **Security Headers**: HSTS, X-Frame-Options, CSP, etc.
- **Health Checks**: Monitor backend health

## Quick Start

### 1. Generate SSL Certificates

**Option A: Let's Encrypt (Recommended for Production)**

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Run certbot to obtain certificates
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email your-email@example.com \
  --agree-tos \
  --no-eff-email \
  -d hrl-finance.yourdomain.com
```

**Option B: Self-Signed Certificates (Development)**

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

### 2. Update Configuration

Edit `nginx/nginx.conf` and update:

```nginx
server_name hrl-finance.yourdomain.com;
```

### 3. Start Services

```bash
# Start with Nginx reverse proxy
docker-compose -f nginx/docker-compose.nginx.yml up -d

# Check status
docker-compose -f nginx/docker-compose.nginx.yml ps

# View logs
docker-compose -f nginx/docker-compose.nginx.yml logs -f nginx
```

### 4. Verify

```bash
# Test HTTP redirect
curl -I http://localhost

# Test HTTPS
curl -k https://localhost/health

# Test WebSocket
wscat -c wss://localhost/socket.io/
```

## Configuration

### Rate Limiting

Adjust rate limits in `nginx.conf`:

```nginx
# API endpoints: 10 requests per second
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# Training endpoints: 1 request per second
limit_req_zone $binary_remote_addr zone=training_limit:10m rate=1r/s;
```

### Caching

Configure cache settings:

```nginx
# Cache path and size
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=100m inactive=60m;

# Cache duration
proxy_cache_valid 200 1h;
```

### Load Balancing

Add more backend servers in `nginx.conf`:

```nginx
upstream hrl_backend {
    server hrl-finance:8000;
    server hrl-finance-2:8000;
    server hrl-finance-3:8000;
    
    least_conn;  # or ip_hash, round_robin
}
```

And uncomment additional services in `docker-compose.nginx.yml`.

### Timeouts

Adjust timeouts for long-running operations:

```nginx
# Standard timeouts
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;

# Training endpoints (longer)
location /api/training/ {
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
}
```

## SSL Certificate Renewal

### Automatic Renewal

The certbot container automatically renews certificates every 12 hours.

### Manual Renewal

```bash
# Renew certificates
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot renew

# Reload Nginx
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -s reload
```

### Test Renewal

```bash
# Dry run
docker-compose -f nginx/docker-compose.nginx.yml run --rm certbot renew --dry-run
```

## Monitoring

### Access Logs

```bash
# View access logs
docker-compose -f nginx/docker-compose.nginx.yml exec nginx tail -f /var/log/nginx/hrl-finance-access.log

# View error logs
docker-compose -f nginx/docker-compose.nginx.yml exec nginx tail -f /var/log/nginx/hrl-finance-error.log
```

### Nginx Status

```bash
# Check configuration
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -t

# Reload configuration
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -s reload

# View Nginx status
docker-compose -f nginx/docker-compose.nginx.yml exec nginx ps aux | grep nginx
```

### Cache Statistics

```bash
# View cache directory
docker-compose -f nginx/docker-compose.nginx.yml exec nginx ls -lh /var/cache/nginx

# Clear cache
docker-compose -f nginx/docker-compose.nginx.yml exec nginx rm -rf /var/cache/nginx/*
```

## Security

### Security Headers

The configuration includes:

- **HSTS**: Force HTTPS for 1 year
- **X-Frame-Options**: Prevent clickjacking
- **X-Content-Type-Options**: Prevent MIME sniffing
- **X-XSS-Protection**: Enable XSS filter
- **Referrer-Policy**: Control referrer information

### Rate Limiting

Protects against:

- API abuse
- DDoS attacks
- Brute force attempts

### SSL Configuration

- TLS 1.2 and 1.3 only
- Strong cipher suites
- Session caching

## Troubleshooting

### SSL Certificate Issues

```bash
# Check certificate
openssl x509 -in nginx/ssl/fullchain.pem -text -noout

# Check certificate expiry
openssl x509 -in nginx/ssl/fullchain.pem -noout -dates

# Test SSL configuration
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -t
```

### Connection Issues

```bash
# Test backend connectivity
docker-compose -f nginx/docker-compose.nginx.yml exec nginx wget -O- http://hrl-finance:8000/health

# Check upstream status
docker-compose -f nginx/docker-compose.nginx.yml exec nginx cat /etc/nginx/conf.d/default.conf | grep upstream
```

### WebSocket Issues

```bash
# Check WebSocket headers
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  https://localhost/socket.io/

# Test with wscat
npm install -g wscat
wscat -c wss://localhost/socket.io/
```

### Rate Limiting Issues

```bash
# Check rate limit zones
docker-compose -f nginx/docker-compose.nginx.yml exec nginx cat /etc/nginx/conf.d/default.conf | grep limit_req_zone

# View rate limit logs
docker-compose -f nginx/docker-compose.nginx.yml logs nginx | grep limiting
```

## Performance Tuning

### Worker Processes

Add to `nginx.conf`:

```nginx
worker_processes auto;
worker_connections 1024;
```

### Buffer Sizes

```nginx
client_body_buffer_size 128k;
client_max_body_size 50m;
proxy_buffer_size 4k;
proxy_buffers 8 4k;
proxy_busy_buffers_size 8k;
```

### Keepalive

```nginx
keepalive_timeout 65;
keepalive_requests 100;
```

## Backup

### Backup SSL Certificates

```bash
# Backup certificates
tar czf ssl-backup-$(date +%Y%m%d).tar.gz nginx/ssl/

# Restore certificates
tar xzf ssl-backup-YYYYMMDD.tar.gz
```

### Backup Configuration

```bash
# Backup Nginx config
cp nginx/nginx.conf nginx/nginx.conf.backup

# Restore configuration
cp nginx/nginx.conf.backup nginx/nginx.conf
docker-compose -f nginx/docker-compose.nginx.yml exec nginx nginx -s reload
```

## Production Checklist

- [ ] SSL certificates configured
- [ ] Domain name configured
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backup strategy in place
- [ ] Load balancing configured (if needed)
- [ ] Cache tuned for workload
- [ ] Firewall rules configured

## Support

For Nginx-specific issues:
- Nginx documentation: https://nginx.org/en/docs/
- SSL Labs test: https://www.ssllabs.com/ssltest/
- Let's Encrypt: https://letsencrypt.org/docs/
