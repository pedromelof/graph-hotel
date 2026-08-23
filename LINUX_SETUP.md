## 🛠️ Install Docker

```bash
sudo apt update && sudo apt install -y git curl ca-certificates

# Install Docker and Docker Compose plugin (if not already installed)
curl -fsSL https://get.docker.com | sudo sh
```

## ⚡ Install UV

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell configuration to load uv into PATH
source ~/.bashrc
```

Verify the installation:

```bash
uv --version
```

---

## 📦 Clone Repository & Configure Environment

1. Clone or copy your repository to the Linux machine:

```bash
git clone <YOUR_GIT_REPO_URL> graph-hotel
cd graph-hotel
```

2. Create your `.env` file from the template:

```bash
cp .env.example .env
```

3. Edit `.env` to configure your keys and passwords:

```bash
nano .env
```

Ensure the Neo4j and OpenAI parameters match your configuration:

```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neohotel123
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-large
```

---

## 🗄️ Start the Neo4j Container

```bash
docker compose up -d
```

## 🐍 Install Python Dependencies

```bash
uv sync
```

## 🚀 Start the API

uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

## 🛡️ Run API as a Background Service (systemd)

1. Create a systemd service file:

```bash
sudo nano /etc/systemd/system/graph-hotel-api.service
```

2. Add the following content (replace `/home/youruser/graph-hotel` and `youruser` with your actual Linux user and path):

```ini
[Unit]
Description=Grafo Hotel FastAPI Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/graph-hotel
ExecStart=/home/youruser/.local/bin/uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
EnvironmentFile=/home/youruser/graph-hotel/.env

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now graph-hotel-api
```

4. Check service status:

```bash
sudo systemctl status graph-hotel-api
```
