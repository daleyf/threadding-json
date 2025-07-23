# Run Demo

1. **Build & start all services**
   
```bash
   docker compose up --build
```

• Distributor → http://localhost:8000
• Analyzers → ports 8001, 8002, 8003

Run: start with **small** but can also use **med**, or add any folder to the *repos* folder

```bash
python3 test.py small --host http://localhost:8000
```
You should see ≈60%/30%/10% distribution in the distributor logs.

Simulate failure & recovery
```bash
docker stop resolver_analyzer_2    # take 30% node offline
docker start resolver_analyzer_2   # it’s back online within 5s
```

High-throughput test

```bash
python3 test.py med --host http://localhost:8000
```

Summarize codebase (Daley's sprinkle of interest)
```bash
python3 summarizer.py "explain the codebase"
```
