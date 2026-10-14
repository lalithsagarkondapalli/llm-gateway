# Running the verification on your Mac (about 15 minutes, all free)

## 1. Install Docker Desktop (5 min)

1. Find your chip: Apple menu → **About This Mac**. It says *Apple M1/M2/M3/M4* (Apple silicon) or *Intel*.
2. Download the matching **Docker Desktop for Mac** from https://www.docker.com/products/docker-desktop/
3. Open the `.dmg`, drag Docker into Applications, then open **Docker** from Applications.
4. Accept the terms. Signing in is optional, so you can skip it. Wait until the bottom-left of the Docker window shows **Engine running**.

## 2. Get a free Groq API key (2 min)

1. Go to https://console.groq.com and sign up (Google sign-in works; no credit card needed).
2. Open **API Keys** → **Create API Key** → name it `llm-gateway` → copy the key (it starts with `gsk_`).
   You can't view it again later, so keep it somewhere safe. **Never commit it to GitHub.**

## 3. Run it (about 8 min, mostly waiting)

Open **Terminal** (Cmd+Space, type "Terminal") and run:

```bash
cd ~/Downloads/llm-gateway          # wherever you unzipped it
bash scripts/verify_mac.sh
```

The first build downloads base images, which takes a few minutes. When asked, paste your Groq key (the text stays hidden as you paste) and press Enter.

To avoid pasting it every time, create a `.env` file. It is already in `.gitignore`, so it won't be committed:

```bash
cp .env.example .env
open -e .env        # put your key after GROQ_API_KEY=, then save
```

## 4. What you get

- `docs/VERIFICATION.md`: PASS/FAIL for each check
- `docs/eval_groq.json`: raw eval results
- `README.md`: the "Routing quality vs cost" section filled in with your real numbers
- `docs/resume_bullet.txt`: a resume bullet based on your actual results

Then commit and push:

```bash
git add -A
git commit -m "Verify Docker stack and add real Groq eval results"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Docker is installed but not running` | Open Docker Desktop and wait for *Engine running* |
| `port 8000, 9090 or 3000 is already in use` | Quit the app using that port, or run `bash scripts/verify_mac.sh --skip-stack` |
| `Groq key check: 401` | Key copied wrong. Create a new one in the Groq console |
| Eval is slow or shows retries | Normal: the free tier allows about 30 requests/min and the script waits instead of failing |
| Anything else | Send `docs/VERIFICATION.md` and the terminal output to Claude |
