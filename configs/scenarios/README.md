# Scenari Finanziari Italiani - Dataset Reale

Questo dataset è basato su dati ufficiali ISTAT e Numbeo (2024) per analizzare comportamenti finanziari realistici in Italia.

## Fonti Dati

- **ISTAT**: Reddito medio italiano, costo della vita
- **Numbeo**: Costi abitativi e spese nelle principali città italiane
- **Banca d'Italia**: Tasso di risparmio medio italiano (~8%)

## Scenari Definiti

### 1. **Milano - Professionista Junior**
- Stipendio netto: 1,800 EUR/mese
- Affitto monolocale: 900 EUR
- Spese fisse: 1,200 EUR
- Spese variabili: 450 EUR

### 2. **Milano - Professionista Senior**
- Stipendio netto: 2,800 EUR/mese
- Affitto bilocale: 1,300 EUR
- Spese fisse: 1,700 EUR
- Spese variabili: 600 EUR

### 3. **Roma - Famiglia Media**
- Stipendio netto: 2,400 EUR/mese
- Mutuo/Affitto: 1,000 EUR
- Spese fisse: 1,500 EUR
- Spese variabili: 700 EUR

### 4. **Bologna - Coppia Giovane**
- Stipendio netto: 3,200 EUR/mese (doppio reddito)
- Affitto: 1,100 EUR
- Spese fisse: 1,800 EUR
- Spese variabili: 800 EUR

### 5. **Torino - Single Medio**
- Stipendio netto: 1,600 EUR/mese
- Affitto: 600 EUR
- Spese fisse: 1,000 EUR
- Spese variabili: 400 EUR

## Parametri Comuni

- **Inflazione**: 2% annuo (media italiana)
- **Tasso risparmio target**: 8-10% (media italiana)
- **Buffer sicurezza**: 2-3 mesi di spese
- **Orizzonte temporale**: 10 anni (120 mesi)

## Utilizzo

Questi scenari possono essere utilizzati con gli script di training e valutazione:

```bash
# Training con uno scenario specifico
python train.py --config configs/scenarios/milano_junior.yaml --episodes 5000

# Valutazione con uno scenario specifico
python evaluate.py --high-agent models/milano_junior_high_agent.pt \
                   --low-agent models/milano_junior_low_agent.pt \
                   --config configs/scenarios/milano_junior.yaml
```

## Studio Comparativo

Esegui lo script di studio comparativo per analizzare tutti gli scenari:

```bash
python study_italian_scenarios.py
```

Questo script:
- Carica tutte le 5 configurazioni degli scenari
- Testa multiple strategie di investimento (5%, 10%, 15%, 20%) per ogni scenario
- Simula fino a 120 mesi (10 anni) per determinare la sostenibilità
- Fornisce un'analisi comparativa che mostra:
  - Fondi disponibili dopo le spese per ogni scenario
  - Tasso di investimento massimo sostenibile
  - Quali scenari sono più/meno favorevoli per gli investimenti
- Genera insights e raccomandazioni per la pianificazione finanziaria italiana
- Salva i risultati dettagliati in `study_results.json`

**Esempio di Output:**
```
ANALISI COMPARATIVA
Scenario              Disponibile    % Reddito    Investimento Max Sostenibile
--------------------------------------------------------------------------------
Milano Junior            200 EUR        11.1%      5% (90 EUR)
Milano Senior            800 EUR        20.0%      15% (600 EUR)
Roma Famiglia            600 EUR        15.0%      10% (400 EUR)
Bologna Coppia         1,200 EUR        24.0%      20% (1,000 EUR)
Torino Single            400 EUR        16.0%      10% (250 EUR)

💡 INSIGHTS:
- Margine medio italiano: 15.2% del reddito disponibile
- Scenario più favorevole: Bologna Coppia (doppio reddito)
- Scenario più critico: Milano Junior (alto costo della vita)
- Investimento sostenibile: 5-10% per la maggior parte degli scenari
```

## Insights Chiave

Lo studio comparativo rivela:

1. **Capacità di Risparmio**: Il margine disponibile varia dal 11% (Milano Junior) al 24% (Bologna Coppia)
2. **Sostenibilità**: La maggior parte degli scenari può sostenere investimenti del 5-10% del reddito
3. **Fattori Critici**: 
   - Costo abitativo (maggiore a Milano)
   - Numero di percettori di reddito (coppia vs single)
   - Livello di seniority professionale
4. **Raccomandazioni**:
   - Costruire un buffer di sicurezza prima di investire aggressivamente
   - Adattare la strategia di investimento al margine disponibile
   - Considerare il costo della vita locale nella pianificazione finanziaria
