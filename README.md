# Prereqs
- Docker & Docker Compose  
- Python 
- Set `OPENAI_API_KEY` in your env  

## 0. **Setup environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. **Build & start all services**
   
```bash
   docker compose up --build
```


## 2. **Run** start with **small** but can also use **med**, or add any folder to the *repos* folder

```bash
python3 test.py small --host http://localhost:8000
```
*You should see ≈60%/30%/10% distribution in the distributor logs.*

## 3 **Simulate failure & recovery**
```bash
docker stop resolver_analyzer_2    # take 30% node offline
docker start resolver_analyzer_2   # it’s back online within 5s
```

## 4 **High-throughput test**

```bash
python3 test.py med --host http://localhost:8000
```

## 5 **Summarize codebase (Daley's sprinkle of interest)**
```bash
python3 summarizer.py "explain the codebase"
```
