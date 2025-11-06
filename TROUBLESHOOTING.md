# Troubleshooting Guide

Common issues and solutions for the HRL Finance System.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Backend Issues](#backend-issues)
3. [Frontend Issues](#frontend-issues)
4. [Training Issues](#training-issues)
5. [Simulation Issues](#simulation-issues)
6. [Performance Issues](#performance-issues)
7. [Data Issues](#data-issues)
8. [Network Issues](#network-issues)
9. [Browser Issues](#browser-issues)
10. [Getting Help](#getting-help)

## Installation Issues

### Python Dependencies Won't Install

**Problem**: `pip install -r requirements.txt` fails

**Solutions**:
1. **Check Python Version**:
   ```bash
   python --version  # Should be 3.10 or higher
   ```
   
2. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install in Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Install Dependencies Individually**:
   ```bash
   pip install fastapi uvicorn pydantic python-socketio pyyaml torch
   ```

5. **Check for System Dependencies**:
   - PyTorch may require specific CUDA versions
   - WeasyPrint requires system libraries (for PDF generation)

### Node Dependencies Won't Install

**Problem**: `npm install` fails in frontend directory

**Solutions**:
1. **Check Node Version**:
   ```bash
   node --version  # Should be 18 or higher
   npm --version   # Should be 9 or higher
   ```

2. **Clear npm Cache**:
   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Use Different Registry**:
   ```bash
   npm install --registry=https://registry.npmjs.org/
   ```

4. **Install with Legacy Peer Deps**:
   ```bash
   npm install --legacy-peer-deps
   ```

### WeasyPrint Installation Fails

**Problem**: PDF report generation not working

**Solutions**:
1. **macOS**:
   ```bash
   brew install python3 cairo pango gdk-pixbuf libffi
   pip install weasyprint
   ```

2. **Ubuntu/Debian**:
   ```bash
   sudo apt-get install python3-dev python3-pip python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
   pip install weasyprint
   ```

3. **Windows**:
   ```bash
   # Download GTK3 runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
   pip install weasyprint
   ```

4. **Fallback**: Use HTML reports instead of PDF

## Backend Issues

### Backend Won't Start

**Problem**: `uvicorn main:socket_app --reload` fails

**Solutions**:
1. **Check if Port is in Use**:
   ```bash
   # macOS/Linux
   lsof -i :8000
   
   # Windows
   netstat -ano | findstr :8000
   ```
   
   Kill the process or use a different port:
   ```bash
   uvicorn main:socket_app --reload --port 8001
   ```

2. **Check Working Directory**:
   ```bash
   cd backend  # Must be in backend directory
   uvicorn main:socket_app --reload
   ```

3. **Check Import Errors**:
   ```bash
   python -c "from backend.main import socket_app"
   ```
   
   If this fails, check that all dependencies are installed.

4. **Check Python Path**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   uvicorn main:socket_app --reload
   ```

### API Returns 500 Internal Server Error

**Problem**: API endpoints return 500 errors

**Solutions**:
1. **Check Backend Logs**:
   - Look at terminal where uvicorn is running
   - Check for Python exceptions and stack traces

2. **Check File Permissions**:
   ```bash
   # Ensure directories are writable
   chmod -R 755 configs/ models/ results/ reports/
   ```

3. **Check Disk Space**:
   ```bash
   df -h  # Ensure sufficient disk space
   ```

4. **Verify File Paths**:
   - Check that `configs/`, `models/`, `results/`, `reports/` directories exist
   - Create missing directories:
     ```bash
     mkdir -p configs/scenarios models results/simulations reports
     ```

5. **Check Database/File Locks**:
   - Ensure no other process is accessing files
   - Restart backend server

### WebSocket Connection Fails

**Problem**: Real-time training updates not working

**Solutions**:
1. **Check Backend is Running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify WebSocket Endpoint**:
   - Should be `ws://localhost:8000/socket.io`
   - Check browser console for connection errors

3. **Check CORS Settings**:
   - Ensure backend allows frontend origin
   - Check `main.py` CORS configuration

4. **Firewall/Proxy Issues**:
   - Disable firewall temporarily to test
   - Check if proxy is blocking WebSocket connections

5. **Use Polling Fallback**:
   - Frontend automatically polls `/api/training/status` every 5 seconds
   - This works even if WebSocket fails

### Training Service Crashes

**Problem**: Training starts but crashes mid-way

**Solutions**:
1. **Check Memory Usage**:
   ```bash
   # Monitor memory while training
   top  # or htop on Linux
   ```
   
   If memory is full:
   - Reduce batch size in training config
   - Reduce number of episodes
   - Close other applications

2. **Check for NaN Values**:
   ```bash
   python debug_nan.py
   ```
   
   If NaN detected:
   - Reduce learning rates
   - Adjust reward coefficients
   - Check scenario parameters

3. **Check Model Save Path**:
   ```bash
   ls -la models/
   # Ensure directory exists and is writable
   ```

4. **Review Training Logs**:
   - Check for error messages in backend logs
   - Look for PyTorch errors or warnings

## Frontend Issues

### Frontend Won't Start

**Problem**: `npm run dev` fails

**Solutions**:
1. **Check Node Modules**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   npm run dev
   ```

2. **Check Port Availability**:
   ```bash
   # Default port is 5173
   lsof -i :5173  # macOS/Linux
   netstat -ano | findstr :5173  # Windows
   ```
   
   Change port in `vite.config.ts` if needed.

3. **Check for Syntax Errors**:
   ```bash
   npm run build
   # Look for TypeScript or build errors
   ```

4. **Clear Vite Cache**:
   ```bash
   rm -rf node_modules/.vite
   npm run dev
   ```

### Page Shows Blank Screen

**Problem**: Frontend loads but shows nothing

**Solutions**:
1. **Check Browser Console**:
   - Open DevTools (F12)
   - Look for JavaScript errors
   - Check Network tab for failed requests

2. **Check Backend Connection**:
   ```bash
   curl http://localhost:8000/health
   ```
   
   If backend is down, start it:
   ```bash
   cd backend
   uvicorn main:socket_app --reload
   ```

3. **Clear Browser Cache**:
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or clear browser cache completely

4. **Check API Base URL**:
   - Verify `src/services/api.ts` has correct backend URL
   - Default should be `http://localhost:8000`

### API Requests Fail with CORS Error

**Problem**: Browser console shows CORS policy errors

**Solutions**:
1. **Check Backend CORS Configuration**:
   - Verify `main.py` includes frontend origin in CORS middleware
   - Default should allow `http://localhost:5173`

2. **Restart Backend**:
   ```bash
   # Stop backend (Ctrl+C)
   # Start again
   uvicorn main:socket_app --reload
   ```

3. **Check Frontend URL**:
   - Ensure frontend is running on expected port
   - Update CORS origins if using different port

4. **Temporary Workaround**:
   - Add `"*"` to allowed origins (development only!)
   - Never use in production

### Charts Not Rendering

**Problem**: Charts show empty or don't appear

**Solutions**:
1. **Check Data Format**:
   - Open browser console
   - Look for data structure errors
   - Verify API response format

2. **Check Recharts Installation**:
   ```bash
   npm list recharts
   # Should show recharts@^2.x.x
   ```
   
   Reinstall if missing:
   ```bash
   npm install recharts
   ```

3. **Check Container Size**:
   - Charts need non-zero width/height
   - Inspect element to verify dimensions

4. **Clear Component State**:
   - Refresh page
   - Navigate away and back
   - Check for state management issues

## Training Issues

### Agent Keeps Going Bankrupt

**Problem**: Training shows agent running out of money

**Solutions**:
1. **Increase Safety Threshold**:
   - Edit scenario: increase `safety_threshold`
   - Recommended: 1-2 months of expenses

2. **Use Conservative Profile**:
   - Start with conservative template
   - Lower risk tolerance (0.2-0.4)

3. **Adjust Reward Coefficients**:
   - Increase `beta` (stability penalty)
   - Decrease `alpha` (investment reward)
   - Example: `beta: 0.5`, `alpha: 5.0`

4. **Increase Initial Cash**:
   - Give agent starting buffer
   - Example: `initial_cash: 5000`

5. **Train Longer**:
   - Agent needs time to learn
   - Try 3000-5000 episodes

### Reward Not Increasing

**Problem**: Training reward stays flat or decreases

**Solutions**:
1. **Train Longer**:
   - Learning can be slow initially
   - Wait for at least 1000 episodes

2. **Adjust Learning Rates**:
   - Increase if learning too slow: `learning_rate_low: 0.001`
   - Decrease if unstable: `learning_rate_low: 0.0001`

3. **Check for NaN Values**:
   ```bash
   python debug_nan.py
   ```
   
   If NaN found:
   - Reduce learning rates
   - Adjust reward scaling
   - Check scenario parameters

4. **Simplify Scenario**:
   - Reduce max_months (e.g., 30 instead of 60)
   - Reduce expense variability
   - Use fixed investment returns

5. **Check Reward Configuration**:
   - Ensure reward coefficients are reasonable
   - Use default values as starting point
   - Avoid extreme values

### Training is Too Slow

**Problem**: Training takes too long

**Solutions**:
1. **Reduce Episodes**:
   - Use 1000 episodes for experiments
   - Increase to 5000 only for final models

2. **Reduce Max Months**:
   - Edit scenario: `max_months: 30` instead of 60
   - Shorter episodes = faster training

3. **Reduce Batch Size**:
   - Edit training config: `batch_size: 16` instead of 32
   - Uses less memory, slightly faster

4. **Close Other Applications**:
   - Free up CPU and memory
   - Close browser tabs
   - Stop other processes

5. **Use GPU** (if available):
   - PyTorch will automatically use GPU if available
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

### Training Stops Unexpectedly

**Problem**: Training stops before completion

**Solutions**:
1. **Check Backend Logs**:
   - Look for error messages
   - Check for exceptions

2. **Check Memory Usage**:
   - Training may crash if out of memory
   - Reduce batch size or episodes

3. **Check Disk Space**:
   - Ensure sufficient space for model checkpoints
   - Each checkpoint is ~10-50 MB

4. **Restart Training**:
   - Models are saved at intervals
   - Can resume from last checkpoint (if implemented)

5. **Check for Timeouts**:
   - Long training may timeout
   - Increase timeout settings if needed

## Simulation Issues

### Simulation Results Look Wrong

**Problem**: Simulation produces unexpected results

**Solutions**:
1. **Check Model Quality**:
   - Review training metrics
   - Ensure training completed successfully
   - Check that stability was >90%

2. **Verify Scenario Parameters**:
   - Ensure scenario matches expectations
   - Check income, expenses, risk tolerance
   - Verify investment return settings

3. **Run More Episodes**:
   - 10 episodes may not be enough
   - Try 20-50 episodes for better statistics

4. **Check Random Seed**:
   - Different seeds produce different results
   - Use same seed for reproducibility
   - Run multiple seeds to verify consistency

5. **Compare with Training**:
   - Simulation results should match training metrics
   - If very different, model may not have learned well

### Simulation Takes Too Long

**Problem**: Simulation runs slowly

**Solutions**:
1. **Reduce Episodes**:
   - Use 10 episodes for quick checks
   - Increase only when needed

2. **Check Backend Performance**:
   - Restart backend server
   - Close other applications
   - Check CPU usage

3. **Simplify Scenario**:
   - Reduce max_months
   - Use fixed investment returns

4. **Check for Infinite Loops**:
   - Look at backend logs
   - Check for error messages

### Simulation Fails to Start

**Problem**: Clicking "Run Simulation" does nothing or errors

**Solutions**:
1. **Check Model Exists**:
   ```bash
   ls -la models/
   # Verify model files exist
   ```

2. **Check Scenario Exists**:
   ```bash
   ls -la configs/scenarios/
   # Verify scenario file exists
   ```

3. **Check Backend Logs**:
   - Look for error messages
   - Check for file not found errors

4. **Verify Model Format**:
   - Model files should be `.pt` (PyTorch)
   - Check file size (should be >1 KB)

5. **Try Different Model/Scenario**:
   - Test with known working combination
   - Isolate the problem

## Performance Issues

### System Runs Slowly

**Problem**: Overall system performance is poor

**Solutions**:
1. **Check System Resources**:
   ```bash
   # Monitor CPU, memory, disk
   top  # or htop
   ```

2. **Close Unnecessary Applications**:
   - Close browser tabs
   - Stop other processes
   - Free up memory

3. **Restart Services**:
   ```bash
   # Stop backend (Ctrl+C)
   # Stop frontend (Ctrl+C)
   # Start both again
   ```

4. **Clear Browser Cache**:
   - Hard refresh: Ctrl+Shift+R
   - Clear all browser data

5. **Check Disk Space**:
   ```bash
   df -h
   # Ensure >1GB free space
   ```

6. **Optimize Database**:
   - Delete old simulation results
   - Remove unused models
   - Clean up reports directory

### Memory Usage Too High

**Problem**: System uses too much memory

**Solutions**:
1. **Reduce Batch Size**:
   - Edit training config: `batch_size: 16`
   - Smaller batches use less memory

2. **Reduce Episodes**:
   - Train with fewer episodes
   - Run fewer simulation episodes

3. **Close Browser Tabs**:
   - Each tab uses memory
   - Keep only necessary tabs open

4. **Restart Backend**:
   - Memory leaks may accumulate
   - Restart periodically

5. **Upgrade System**:
   - Minimum 4GB RAM recommended
   - 8GB+ for better performance

### Disk Space Running Out

**Problem**: Running out of disk space

**Solutions**:
1. **Clean Up Old Files**:
   ```bash
   # Remove old simulation results
   rm results/simulations/*.json
   
   # Remove old reports
   rm reports/*.html reports/*.pdf
   
   # Remove old model checkpoints
   rm -rf models/checkpoints/
   ```

2. **Archive Important Data**:
   - Export important results
   - Move to external storage
   - Keep only recent data

3. **Reduce Checkpoint Frequency**:
   - Increase save_interval in training
   - Example: `save_interval: 500` instead of 100

4. **Compress Old Files**:
   ```bash
   tar -czf archive.tar.gz results/ reports/
   rm -rf results/ reports/
   ```

## Data Issues

### Scenarios Not Loading

**Problem**: Scenario list is empty or won't load

**Solutions**:
1. **Check Scenarios Directory**:
   ```bash
   ls -la configs/scenarios/
   # Should contain .yaml files
   ```

2. **Create Test Scenario**:
   - Use the web interface to create a new scenario
   - Or copy from templates:
     ```bash
     cp configs/balanced.yaml configs/scenarios/test.yaml
     ```

3. **Check File Permissions**:
   ```bash
   chmod 644 configs/scenarios/*.yaml
   ```

4. **Verify YAML Format**:
   - Open scenario file in text editor
   - Check for syntax errors
   - Validate YAML: https://www.yamllint.com/

5. **Check Backend Logs**:
   - Look for file reading errors
   - Check for permission denied errors

### Models Not Appearing

**Problem**: Model list is empty

**Solutions**:
1. **Check Models Directory**:
   ```bash
   ls -la models/
   # Should contain *_high_agent.pt and *_low_agent.pt files
   ```

2. **Complete a Training**:
   - Train a model to completion
   - Check that files are saved

3. **Check File Naming**:
   - Files should follow pattern: `{name}_high_agent.pt`, `{name}_low_agent.pt`
   - History file: `{name}_history.json`

4. **Check File Permissions**:
   ```bash
   chmod 644 models/*.pt models/*.json
   ```

5. **Verify Model Format**:
   - Files should be PyTorch format (.pt)
   - Check file size (should be >1 KB)

### Simulation Results Missing

**Problem**: Can't find simulation results

**Solutions**:
1. **Check Results Directory**:
   ```bash
   ls -la results/simulations/
   # Should contain .json files
   ```

2. **Run a Simulation**:
   - Complete a simulation
   - Check that results are saved

3. **Check File Naming**:
   - Files follow pattern: `{model}_{scenario}_{timestamp}.json`

4. **Check File Permissions**:
   ```bash
   chmod 644 results/simulations/*.json
   ```

5. **Verify JSON Format**:
   - Open file in text editor
   - Check for valid JSON syntax

### Data Corruption

**Problem**: Files are corrupted or unreadable

**Solutions**:
1. **Restore from Backup**:
   - If you have backups, restore them
   - Always keep backups of important data

2. **Recreate Data**:
   - Recreate scenarios from scratch
   - Retrain models
   - Rerun simulations

3. **Check Disk Health**:
   ```bash
   # macOS
   diskutil verifyVolume /
   
   # Linux
   sudo fsck /dev/sda1
   ```

4. **Prevent Future Corruption**:
   - Don't force quit during saves
   - Ensure sufficient disk space
   - Regular backups

## Network Issues

### Cannot Connect to Backend

**Problem**: Frontend can't reach backend API

**Solutions**:
1. **Verify Backend is Running**:
   ```bash
   curl http://localhost:8000/health
   ```
   
   Should return: `{"status": "healthy", "timestamp": "..."}`

2. **Check Backend URL**:
   - Frontend default: `http://localhost:8000`
   - Verify in `src/services/api.ts`

3. **Check Firewall**:
   - Temporarily disable firewall
   - Add exception for port 8000

4. **Check Network**:
   ```bash
   ping localhost
   # Should respond
   ```

5. **Try Different Port**:
   - Start backend on different port
   - Update frontend API URL

### WebSocket Keeps Disconnecting

**Problem**: Real-time updates stop working

**Solutions**:
1. **Check Network Stability**:
   - Ensure stable internet connection
   - Check for network interruptions

2. **Check Backend Logs**:
   - Look for WebSocket errors
   - Check for connection timeouts

3. **Increase Timeout**:
   - Modify WebSocket timeout settings
   - In `src/services/websocket.ts`

4. **Use Polling Fallback**:
   - Frontend automatically falls back to polling
   - Check `/api/training/status` endpoint

5. **Restart Both Services**:
   - Stop backend and frontend
   - Start both again

### Requests Timeout

**Problem**: API requests take too long and timeout

**Solutions**:
1. **Check Backend Performance**:
   - Monitor CPU and memory usage
   - Restart backend if needed

2. **Increase Timeout**:
   - Modify axios timeout in `src/services/api.ts`
   - Default is usually 30 seconds

3. **Reduce Request Size**:
   - Limit number of episodes
   - Reduce data being transferred

4. **Check Network Speed**:
   - Test network connection
   - Use wired connection if possible

5. **Optimize Backend**:
   - Close other applications
   - Free up system resources

## Browser Issues

### Browser Console Shows Errors

**Problem**: JavaScript errors in browser console

**Solutions**:
1. **Clear Browser Cache**:
   - Hard refresh: Ctrl+Shift+R
   - Clear all browser data

2. **Update Browser**:
   - Use latest version of Chrome, Firefox, Safari, or Edge
   - Older browsers may not support modern features

3. **Disable Extensions**:
   - Try in incognito/private mode
   - Disable ad blockers and extensions

4. **Check for TypeScript Errors**:
   ```bash
   cd frontend
   npm run build
   # Look for compilation errors
   ```

5. **Restart Browser**:
   - Close all browser windows
   - Restart browser

### Dark Mode Not Working

**Problem**: Theme toggle doesn't work

**Solutions**:
1. **Clear Local Storage**:
   - Open DevTools (F12)
   - Application tab → Local Storage
   - Clear all data

2. **Check Theme Context**:
   - Verify ThemeProvider is wrapping app
   - Check `src/App.tsx`

3. **Hard Refresh**:
   - Ctrl+Shift+R (Windows/Linux)
   - Cmd+Shift+R (Mac)

4. **Check CSS**:
   - Verify Tailwind dark mode classes
   - Check `tailwind.config.js`

### Mobile View Issues

**Problem**: Interface doesn't work well on mobile

**Solutions**:
1. **Use Desktop Browser**:
   - System is optimized for desktop
   - Mobile support is basic

2. **Rotate Device**:
   - Use landscape orientation
   - More screen space available

3. **Zoom Out**:
   - Pinch to zoom out
   - See more content at once

4. **Use Responsive Features**:
   - Hamburger menu for navigation
   - Scrollable tables and charts

## Getting Help

### Before Asking for Help

1. **Check This Guide**: Review relevant sections above
2. **Check Documentation**: See README.md, USER_GUIDE.md
3. **Check Logs**: Review backend and browser console logs
4. **Try Basic Fixes**: Restart services, clear cache, refresh page
5. **Isolate Problem**: Determine if issue is backend, frontend, or data

### Information to Provide

When asking for help, include:

1. **System Information**:
   - Operating system and version
   - Python version: `python --version`
   - Node version: `node --version`
   - Browser and version

2. **Error Messages**:
   - Complete error message
   - Stack trace if available
   - Browser console errors

3. **Steps to Reproduce**:
   - What you were trying to do
   - Steps taken before error
   - Expected vs actual behavior

4. **Logs**:
   - Backend terminal output
   - Browser console output
   - Any relevant log files

5. **Configuration**:
   - Scenario parameters
   - Training configuration
   - Any custom settings

### Where to Get Help

1. **Documentation**:
   - README.md - Complete system documentation
   - USER_GUIDE.md - User interface guide
   - QUICK_START.md - Getting started guide
   - API Documentation - http://localhost:8000/docs

2. **Examples**:
   - examples/ directory - Working code examples
   - tests/ directory - Test cases showing usage

3. **Community**:
   - GitHub Issues - Report bugs and request features
   - Discussions - Ask questions and share ideas

### Debug Mode

Enable debug mode for more detailed logging:

1. **Backend Debug Mode**:
   ```bash
   # Set log level to DEBUG
   export LOG_LEVEL=DEBUG
   uvicorn main:socket_app --reload --log-level debug
   ```

2. **Frontend Debug Mode**:
   - Open browser DevTools (F12)
   - Check Console, Network, and Application tabs
   - Enable verbose logging in browser settings

3. **Python Debug Mode**:
   ```python
   # Add to code for debugging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Common Error Messages

**"ModuleNotFoundError: No module named 'X'"**:
- Solution: `pip install X`

**"EADDRINUSE: address already in use"**:
- Solution: Kill process using port or use different port

**"CORS policy: No 'Access-Control-Allow-Origin' header"**:
- Solution: Check backend CORS configuration

**"Cannot read property 'X' of undefined"**:
- Solution: Check data structure, add null checks

**"Failed to fetch"**:
- Solution: Check backend is running, verify URL

**"WebSocket connection failed"**:
- Solution: Check backend WebSocket endpoint, try polling fallback

---

## Still Having Issues?

If you've tried everything and still have problems:

1. **Create Minimal Reproduction**:
   - Simplify to smallest case that shows problem
   - Use default configurations
   - Test with fresh installation

2. **Check System Requirements**:
   - Python 3.10+
   - Node 18+
   - 4GB+ RAM
   - 1GB+ free disk space

3. **Try Fresh Installation**:
   ```bash
   # Backup your data first!
   rm -rf venv node_modules
   # Reinstall everything
   ```

4. **Report Bug**:
   - Open GitHub issue
   - Include all information listed above
   - Provide minimal reproduction steps

---

**Last Updated**: November 2024
**Version**: 1.0
**For**: HRL Finance System
