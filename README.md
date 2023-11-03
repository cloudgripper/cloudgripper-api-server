# CloudGripper API

### Set environment variables using your credentials
```
export MONGO_CLIENT_USERNAME='username'
export MONGO_CLIENT_PASSWORD='password'
export JWT_SECRET_KEY='secret-key'
```

### Cloning the CloudGripper API project
```
mkdir -p ~/cloudgripper/
cd ~/cloudgripper/
git clone https://github.com/cloudgripper/cloudgripper-api-server.git
```

### Creating and sourse virtual environment
```
python3 -m venv --system-site-packages .env
source .env/bin/activate
```

### Install requirements
```
pip install -r requirements.txt
sudo apt-get install -y python3-picamera2
sudo apt-get install -y ffmpeg
```

### Running the API server
```
cd ~/cloudgripper/cloudgripper-api-server/
python3 app.py
```

