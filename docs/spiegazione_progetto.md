# Spiegazione del progetto: da v0 a oggi

Appunti personali. Pensati per chi viene da ML classico (regressione, classificazione) e non ha mai fatto information retrieval.

---

## 1. Il problema di partenza

Un LLM riceve un **prompt**. Il prompt ha una lunghezza massima, misurata in **token**. Un token è un pezzo di parola: in inglese, circa 0,75 parole per token. Più token metti nel prompt, più l'LLM costa ed è lento, e spesso risponde peggio.

Un sistema **RAG** (Retrieval-Augmented Generation) funziona così:

1. arriva una domanda dell'utente (la **query**);
2. si cercano in un archivio dei documenti che potrebbero contenere la risposta. Questa è la fase di **retrieval** (recupero);
3. alcuni di quei documenti vengono incollati nel prompt insieme alla domanda;
4. l'LLM risponde usando quei documenti.

Questo progetto riguarda **solo il passo 3**. Hai N documenti candidati (negli esperimenti N = 25) e un budget massimo di token W_max. Devi scegliere **quali** documenti mettere nel prompt.

---

## 2. Come si misura un documento

**Embedding.** Una rete neurale (sentence-transformers) trasforma un testo in un vettore di numeri, per esempio di 384 dimensioni. È un vettore di feature. Testi con significato simile danno vettori che puntano in direzioni simili.

**Cosine similarity.** È il coseno dell'angolo tra due vettori. Vale 1 se puntano nella stessa direzione e 0 se sono ortogonali. Serve per due cose:

- **rilevanza** r_i: similarità tra la query e il documento i. Dice quanto il documento c'entra con la domanda;
- **ridondanza** sim(i,j): similarità tra due documenti. Se è alta, dicono più o meno la stessa cosa.

**Costo** w_i: il numero di token del documento i.

---

## 3. La funzione obiettivo

Dato un sottoinsieme S di documenti:

```
Score(S) = Σ_{i ∈ S} r_i  −  λ · Σ_{i<j, i,j ∈ S} sim(i,j)

vincolo:   Σ_{i ∈ S} w_i  ≤  W_max
```

- La prima parte premia i documenti rilevanti.
- La seconda punisce il mettere insieme documenti che si ripetono.
- λ = 0,1 regola quanto pesa la punizione.

In ottimizzazione, scegliere oggetti con un valore e un peso sotto una capacità massima si chiama **knapsack problem**. Qui il valore non è una semplice somma, perché c'è il termine sulle coppie. Per questo è un **Quadratic Knapsack Problem**, che è NP-hard: trovare la soluzione migliore esatta diventa rapidamente costoso al crescere di N.

---

## 4. I metodi confrontati

Tutti prendono in input gli stessi r, sim, w e W_max, e restituiscono un sottoinsieme S.

| Metodo | Come sceglie |
|---|---|
| **top_k** | Ordina per rilevanza r e aggiunge documenti finché il budget lo permette. Ignora ridondanza e lunghezza. |
| **MMR** (Maximal Marginal Relevance) | Metodo classico del retrieval. A ogni passo aggiunge il documento con il miglior compromesso tra rilevanza e differenza da quelli già scelti. |
| **greedy_objective** | A ogni passo calcola il **guadagno marginale** Δ_i: di quanto sale Score se aggiungo i? Poi aggiunge il documento con Δ_i più grande. |
| **greedy_token_aware** | Uguale, ma sceglie il documento con Δ_i / w_i più grande, cioè il guadagno **per token**. |
| **ILP** (Integer Linear Programming) | Un solver matematico esatto (PuLP/CBC). Trova la soluzione **ottima vera**, che chiamiamo **OPT**. È lento, ma su N = 25 funziona. |

Il guadagno marginale di aggiungere i all'insieme corrente S è:

```
Δ_i = r_i − λ · Σ_{j ∈ S} sim(i,j)
```

I due greedy si fermano quando nessun documento rimasto entra nel budget, oppure quando nessuno ha guadagno positivo.

**Gap** = OPT − Score(metodo). È quanto un metodo perde rispetto all'ottimo. Svolge il ruolo della "ground truth" nel ML supervisionato: sai qual è la risposta giusta e misuri l'errore.

---

## 5. v0: cosa c'era all'inizio

Il README sosteneva che il metodo token-aware fosse migliore. L'audit ha trovato problemi nel codice:

- il solver ILP a volte non trovava l'ottimo, ma il codice lo trattava comunque come ottimo;
- l'"MMR" implementato non era il vero MMR;
- i metodi usavano regole di stop diverse, quindi il confronto non era equo;
- alcune metriche (densità, latenza) erano calcolate o descritte in modo fuorviante.

In più c'era un solo run su pochi dati, senza statistica: nessuna ripetizione, nessun intervallo di confidenza.

v0 è stato conservato così com'era, come riferimento storico.

---

## 6. Step 1–2: riparazione e rivalutazione

- **Step 1.** Ho corretto il codice: il solver ora dichiara se ha davvero dimostrato l'ottimo, l'MMR è quello vero e la regola di stop è comune a tutti. Ho aggiunto 44 test.
- **Step 2.** Ho rifatto il benchmark originale con molti **seed** e con statistica vera.

**Cos'è un seed qui.** Ogni seed genera un problema casuale diverso: documenti, rilevanze e lunghezze. Un seed equivale a **un'osservazione indipendente** nel dataset dell'esperimento. 60 seed significano 60 problemi indipendenti.

---

## 7. Step 3–5: il benchmark controllato

**Problema.** Con i dati reali non puoi controllare le proprietà dei documenti. Per esempio, non puoi decidere se i documenti lunghi siano più o meno rilevanti di quelli corti.

**Soluzione.** Un **generatore sintetico**: si creano problemi artificiali in cui queste proprietà sono dei parametri. È un **disegno sperimentale**: variabili indipendenti controllate, una variabile risposta misurata.

- **Step 3.** Ho costruito il generatore, poi fatto un pilota e un pre-flight per verificare che generasse davvero ciò che dichiarava.
- **Step 4.** L'esperimento completo: 5400 righe di risultati.
- **Step 5.** Un audit statistico dei risultati. Ha scoperto che una conclusione precedente ("nessuna evidenza che token-aware aiuti") dipendeva da **quali budget** erano stati scelti, non da un vero effetto nullo.

Lezione: **l'effetto dipende fortemente dal budget.** Da qui nasce l'esperimento successivo.

---

## 8. Step 6: l'esperimento di "elasticità"

### 8.1 La domanda

Dividere per i token (Δ_i / w_i invece di Δ_i) cambia la scelta **solo se** l'ordinamento per rilevanza è diverso dall'ordinamento per rilevanza per token.

- Se i documenti più rilevanti sono anche i più corti, i due ordinamenti coincidono, quindi i due greedy scelgono le stesse cose.
- Se i documenti più rilevanti sono i più lunghi, i due ordinamenti divergono, quindi i due greedy scelgono cose diverse.

**Domanda:** come cambia il vantaggio di token-aware al variare della **correlazione tra rilevanza e lunghezza**, e al variare del budget?

### 8.2 Il disegno (in termini ML/statistici)

**Fattori controllati:**

- **β**: la correlazione tra rilevanza e lunghezza, con 6 livelli da −0,5 a 2. Si genera con una **copula gaussiana**, una tecnica per produrre due variabili con distribuzioni marginali fissate e una dipendenza scelta a piacere. Così cambia solo la dipendenza, non la distribuzione della rilevanza o della lunghezza;
- **budget**: 3 livelli, tight / medium / loose, pari a 0,25 / 0,60 / 1,50 × W_sat. W_sat è circa il budget oltre il quale il vincolo sui token smette di essere stringente. "Loose" vuol dire che il budget è quasi abbondante;
- **ridondanza** γ: 2 livelli.

**Misura di divergenza D** = 1 − Kendall τ tra l'ordinamento per r e l'ordinamento per r/w. Kendall τ è una correlazione tra ranking: vale 1 se i due ordinamenti sono identici. D = 0 vuol dire ordinamenti uguali; D grande vuol dire molto diversi.

**Variabile risposta (estimando):**

```
x = Score(token_aware) − Score(objective)
```

x > 0 vuol dire che token-aware ha fatto meglio. Spesso **x = 0 esattamente**, perché i due metodi scelgono lo stesso insieme. È una variabile **zero-inflated**: una massa di zeri più una parte continua.

**Struttura dei dati.** Ogni seed viene valutato in tutte le combinazioni di β, budget e ridondanza. Sono **misure ripetute sulla stessa unità**, come un disegno appaiato o a blocchi.

### 8.3 I test (9, decisi in anticipo)

| Test | Cosa fa |
|---|---|
| **Friedman** | Versione non parametrica, basata sui ranghi, dell'ANOVA a misure ripetute. Chiede se x cambia al variare di β. |
| **Contrasti** | Confronti specifici, per esempio il β più basso contro il più alto. |
| **T8** | Chiede se la **probabilità** che i due metodi scelgano insiemi diversi cresce con D. È una pendenza, come in una regressione. |
| **Sign-flip permutation test** | Test non parametrico per dati appaiati. Sotto l'ipotesi nulla, il segno di x è casuale: si ribaltano i segni a caso 10.000 volte per costruire la distribuzione nulla. |
| **Holm** | Correzione per test multipli, come Bonferroni ma meno conservativa. Con 9 test, qualcuno risulta "significativo" per caso se non correggi. |

### 8.4 Pre-registrazione e "lock"

Tutto (ipotesi, test, soglie, numero di seed) viene **fissato prima di vedere i dati** e "bloccato", cioè salvato con un hash crittografico che dimostra che non è stato cambiato dopo.

Serve a evitare il **p-hacking**: provare tante analisi e riportare quella che viene significativa. v6.0 → v6.1 → v6.2 sono le versioni successive del protocollo. Ogni modifica è stata un "amendment" documentato. **v6.2 è la versione bloccata.**

---

## 9. Il numero di seed: power analysis

Quanti seed servono? Dipende da due cose:

1. **quanto è grande l'effetto che vuoi rilevare** (δ);
2. **quanto sono variabili i dati** (σ²).

Più rumore o effetto più piccolo richiedono più seed. È la power analysis classica.

### 9.1 SESOI

**SESOI** (Smallest Effect Size Of Interest) è la differenza più piccola che ritieni importante nella pratica. Nel protocollo v6.2:

```
δ = 3% di OPT     (fissato separatamente per ogni budget)
```

In parole: "mi interessa se token-aware guadagna almeno il 3% dello score ottimo".

### 9.2 Come si stima la varianza

Serve un **pilota**: un piccolo esperimento su seed separati (2000–2039, 40 seed) usato **solo** per stimare la variabilità, non per testare ipotesi. Da lì si ricavano:

- **π** = frazione di casi con x ≠ 0;
- **M2** = media di x² quando x ≠ 0;
- quindi **π · M2 = E[x²]**, il momento secondo totale.

La varianza prevista, supponendo che la media vera sia δ, è:

```
Var = E[x²] − δ²
```

(più una correzione per la correlazione tra ridondanze).

### 9.3 Lo STOP

Una variabile con media δ ha sempre E[x²] ≥ δ², perché Var = E[x²] − (media)² ≥ 0.

Se i dati mostrano **E[x²] < δ²**, vuol dire che **le x osservate sono tipicamente più piccole di δ**. Chiedere di rilevare una media pari a δ è incompatibile con la scala dei dati: la varianza verrebbe negativa, che è impossibile.

Il protocollo prevedeva questo caso (regola a): **STOP, non calcolare n.**

**Cosa è successo:**

| Budget | RMS(x) / δ | Esito |
|---|---|---|
| tight | 2,66 | ok |
| medium | 1,27 | ok |
| loose | **0,36** | **STOP** |

RMS(x) = √E[x²] è la dimensione tipica di x. A budget loose, la differenza tipica tra i due metodi è circa un terzo del 3% di OPT. Con budget quasi abbondante, entrambi i metodi mettono dentro quasi tutto ciò che conviene, quindi scelgono quasi le stesse cose.

L'audit forense ha verificato che **non è un bug**: la matematica e il codice concordano. È la regola che ha funzionato come doveva.

---

## 10. Perché non basta abbassare δ

La tentazione ovvia: "metto δ = 1% e riparto". Il problema è che lo decideresti **dopo aver visto che il 3% non funziona**, cioè scegli la soglia in base ai dati. È proprio il tipo di scelta che la pre-registrazione vuole impedire. Un δ scelto così non dice più "differenza importante", dice "differenza che riesco a rilevare".

Inoltre si è scoperto che il 3% **non aveva una giustificazione vera**. Era stato preso per analogia da una frase descrittiva dello Step 4 ("≈1% trascurabile, 3,4% il più piccolo chiamato reale"), su dati con una scala diversa.

---

## 11. Dove siamo adesso

- **v6.2** è bloccata e ferma su uno STOP valido. Non si tocca.
- **v6.3** richiede prima una **decisione scientifica tua, non statistica**: *quanto deve essere grande la differenza tra i due metodi perché conti davvero per chi usa il sistema?*
- La checklist ti chiede di rispondere con una **giustificazione esterna**: letteratura, una convenzione del dominio, oppure una misura di conseguenze reali. Non con i numeri del pilota.
- Tutto il resto (varianza, numero di seed, test) segue meccanicamente una volta fissata quella soglia.

**Il cancello:** se non sai giustificare la soglia, l'esperimento confermativo non parte. Le strade legittime sono due:

- **A.** Trovi una giustificazione esterna per la soglia.
- **B.** Cambi domanda e misuri un effetto concreto a valle. Per esempio: le risposte dell'LLM migliorano con il contesto scelto da token-aware? In quel modo stabilisci prima cosa è "importante".

---

## 12. In sintesi

Tutta la parte statistica serve a rispondere a una sola domanda: **"dividere il guadagno per i token migliora la selezione dei documenti, e quando?"**

Cosa si vede già, senza test formali:

- a budget stretto, i due metodi differiscono spesso e di più;
- a budget largo, differiscono poco.

Cosa manca è sapere se quelle differenze **contano per l'uso reale**. Questa domanda non la può chiudere nessun test statistico: serve un criterio esterno ai dati.
