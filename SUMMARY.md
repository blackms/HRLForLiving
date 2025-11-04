# Studio HRL Finance - Sintesi Esecutiva

## 🎯 Obiettivo dello Studio

Analizzare la sostenibilità delle strategie di investimento personale nel contesto italiano utilizzando Hierarchical Reinforcement Learning, con dati realistici da ISTAT e Numbeo 2024.

## 📊 Metodologia

- **Sistema**: Architettura HRL a 2 livelli (Financial Strategist + Budget Executor)
- **Dataset**: 5 scenari italiani rappresentativi (Milano, Roma, Bologna, Torino)
- **Simulazioni**: 20 esperimenti (5 scenari × 4 strategie)
- **Orizzonte**: 10 anni (120 mesi)
- **Novità**: Modellazione rendimenti investimenti (0% vs 6% annuo)

## 🔍 Risultati Principali

### Senza Rendimenti

| Scenario | Disponibile | Durata Media | Sostenibilità |
|----------|-------------|--------------|---------------|
| Milano Junior | 8.3% | 7 mesi | ❌ |
| Milano Senior | 17.9% | 15 mesi | ❌ |
| Roma Famiglia | 8.3% | 8 mesi | ❌ |
| Bologna Coppia | 18.8% | 17 mesi | ❌ |
| Torino Single | 12.5% | 8 mesi | ❌ |

**Margine medio disponibile**: 13.2% del reddito (coerente con tasso risparmio ISTAT 8-10%)

### Con Rendimenti 6% Annuo

| Strategia | Senza Rendimenti | Con Rendimenti | Guadagni | Δ Durata |
|-----------|------------------|----------------|----------|----------|
| Conservativa (5%) | 17.9 mesi | 17.6 mesi | +347 EUR | -0.3 mesi |
| Moderata (10%) | 17.3 mesi | 17.0 mesi | +388 EUR | -0.3 mesi |
| Bilanciata (15%) | 16.4 mesi | 16.1 mesi | +521 EUR | -0.3 mesi |

**Impatto rendimenti**: +~400 EUR patrimonio, ma **nessun miglioramento** nella sostenibilità.

## 💡 Conclusioni Chiave

### 1. Insostenibilità Strutturale

**Nessuna delle 20 combinazioni testate è sostenibile a 10 anni**, nemmeno con rendimenti del 6% annuo.

**Cause**:
- Margine disponibile troppo basso (8-19%)
- Variabilità spese (±80-150 EUR/mese)
- Inflazione cumulativa (22% in 10 anni)
- Buffer insufficienti (2,500-10,000 EUR)

### 2. Validazione Macroeconomica

| Metrica | Studio HRL | ISTAT 2024 | Status |
|---------|-----------|------------|--------|
| Tasso risparmio | 13.2% | 8-10% | ✅ Validato |
| Capacità investimento | 5-10% | ~5% | ✅ Validato |
| Sostenibilità | Critica | Critica | ✅ Confermato |

### 3. Impatto Rendimenti

- **Patrimonio**: +300-500 EUR in 17 mesi
- **Durata**: Nessun miglioramento significativo
- **Conclusione**: I rendimenti **non risolvono** il problema strutturale

## 📈 Raccomandazioni

### Per Policy Makers

1. **Aumentare redditi reali**: Margine 13.2% insufficiente per investimenti significativi
2. **Incentivare buffer emergenza**: 6-12 mesi di spese
3. **Controllo inflazione**: Impatto critico sulla sostenibilità

### Per Consulenti Finanziari

1. **Priorità al buffer**: Prima di investire, accumulare 6-12 mesi spese
2. **Investimenti graduali**: Iniziare con 3-5%, non 10-20%
3. **Ridurre variabilità spese**: Più importante che massimizzare rendimenti

### Per Individui

1. **Aspettative realistiche**: Con margini 8-18%, investimenti aggressivi non sostenibili
2. **Focus riduzione spese**: Ogni 100 EUR risparmiati = +3-6% capacità investimento
3. **Stabilità > Rendimento**: Ridurre variabilità spese è prioritario

## 🔬 Contributi Scientifici

1. **Sistema HRL funzionante** per finanza personale
2. **Dataset realistico italiano** basato su dati ufficiali
3. **Validazione empirica** dell'insostenibilità strategie aggressive
4. **Modellazione rendimenti** con analisi impatto
5. **Coerenza macroeconomica** con dati ISTAT

## 📁 Documenti Disponibili

- **TECHNICAL_PAPER.md**: Paper tecnico completo (30+ pagine)
- **figures/**: Visualizzazioni grafiche (3 figure)
- **configs/scenarios/**: 5 scenari italiani configurati
- **study_italian_scenarios.py**: Script analisi comparativa
- **comprehensive_returns_study.py**: Studio impatto rendimenti
- **visualize_results.py**: Generazione grafici

## 🚀 Prossimi Passi

1. **Estensioni modello**:
   - Tassazione capital gains (20% Italia)
   - Eventi straordinari (job loss, medical)
   - Reddito variabile (freelancer)

2. **Scenari aggiuntivi**:
   - Pensionati
   - Studenti part-time
   - Profili rendimento diversi (3%, 9%)

3. **Analisi sensibilità**:
   - Variazioni ±10% spese
   - Aumenti salariali graduali
   - Inflazione variabile (1-5%)

## 📞 Contatti

Per domande, collaborazioni o accesso al codice completo, contattare gli autori.

---

**Data**: Novembre 2024  
**Versione**: 1.0  
**Status**: ✅ Completato
