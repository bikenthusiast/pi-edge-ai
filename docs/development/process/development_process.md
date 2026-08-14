Initial:

```bash
make setup                                    # Mac: venv + Dev-Tools
make models                                   # Gewichte laden
make test                                     # 21 Tests, keine Hardware nötig
git add -A && git commit -m "initial" && git push

make deploy-dry                               # Filter prüfen
make deploy                                   # rsync zum Pi
ssh pi 'cd pi-edge-ai && bash scripts/bootstrap_pi.sh'
make run-pi                                   # Gegenprobe
```

Regular:

### Mac loop
1. Write test uder test/test_*.py
2. Fail test via *make test*
3. Implement code under src/edge
4. Pass test via *make test && make lint*

###Pi Loop
1. check filter via *make deploy-dry*
2. Deploy via *make deploy*
3. Run test via *make test-pi*
4. test live via *make pipeline*

```bash
# Code auf dem Mac ändern, in VS Code
make test        # schnell, lokal
make deploy      # testet und synchronisiert
make run-pi      # auf echter Hardware verifizieren
git push
```
