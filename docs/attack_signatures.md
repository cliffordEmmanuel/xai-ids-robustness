# What Makes a Flow "an Attack" — CICIDS2017 Signatures

How the different attack families in CICIDS2017 show up in the flow features, and
why that matters for the XAI/tamper study.

**Core idea.** A flow is attack-like when its statistics deviate from the patterns
of normal human/application traffic in ways specific to the attack's mechanism. No
single feature decides it — the model learns *combinations*. The groupings below
are the intuition; feature names refer to `docs/data_dictionary.md`.

**Primary source.** Sharafaldin, Lashkari & Ghorbani, *Toward Generating a New
Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018
([DOI](https://doi.org/10.5220/0006639801080116) ·
[PDF](https://www.scitepress.org/Papers/2018/66398/66398.pdf)).

---

## By attack family

### Volumetric DoS / DDoS (Hulk, GoldenEye, DDoS LOIT)
Signature is **rate and volume**: very high `Flow Packets/s` and `Flow Bytes/s`,
many packets (`Total Fwd Packets`) sent back-to-back so `Flow IAT Mean/Min` is
tiny, often many small packets, and elevated `SYN Flag Count` for SYN floods.
Many flows hit the same target.

### Slow DoS (slowloris, Slowhttptest)
The **opposite** profile, which is what makes it sneaky: long `Flow Duration`,
large `Flow IAT` and `Idle` times, and *few* packets — connections held open with
trickled, incomplete requests. "Long-lived but nearly silent."

### Port scan
Many very short flows with almost no payload (`Total Length of Fwd Packets` ≈ 0),
1–2 packets each, short `Flow Duration`, incomplete handshakes (SYN with no ACK, or
`RST`). The intent — probe many ports — shows as huge numbers of tiny, similar
flows.

### Brute force (FTP-Patator, SSH-Patator)
Many **repeated** connections to the same service/port (22, 21), regular timing,
and clustered packet sizes (each login attempt looks alike). Repetition and
uniformity matter more than any single flow.

### Heartbleed
A request/response **size mismatch**: a small forward request pulls back an
abnormally large response (leaked server memory), so `Total Length of Bwd Packets`
and `Bwd Packet Length Max` are high relative to the forward direction — a skewed
`Down/Up Ratio`.

### Botnet (ARES)
**Periodic beaconing** to a command-and-control host: regular, low-volume flows at
fixed intervals (consistent `Flow IAT`), small uniform packet sizes, repeated to
the same external address. Regularity is the fingerprint.

### Web attacks (XSS, SQL Injection, brute force) & Infiltration
The **hardest** — close to normal HTTP or internal traffic. Signals are subtle:
slightly unusual forward payload sizes (injection strings), anomalous request
patterns, or a normally-quiet internal host suddenly making odd outbound flows.
Low separability, which is why detectors struggle here.

---

## Feature groups → attacks

| Feature group | Example features | Flags these attacks |
|---|---|---|
| Rate | `Flow Bytes/s`, `Flow Packets/s`, `Fwd/Bwd Packets/s` | Volumetric DoS/DDoS |
| Timing / IAT / idle | `Flow IAT *`, `Active *`, `Idle *` | Slow DoS, botnet beaconing |
| Packet counts & lengths | `Total Fwd/Bwd Packets`, `*Packet Length*` | Scans (tiny/many), floods (many/small) |
| TCP flags | `SYN/RST/ACK/FIN Flag Count` | SYN floods, scans, incomplete handshakes |
| Direction ratios | `Down/Up Ratio`, fwd/bwd byte balance | Heartbleed, exfiltration |

---

## Caveats (worth a sentence in the thesis)

- **Destination Port leakage.** The raw `Destination Port` almost perfectly encodes
  the attacked service, giving artificially easy separability — the loader drops it
  by default so the model learns behaviour, not service identity.
- **Synthetic timing artefacts.** Some separability comes from how the traffic was
  generated rather than intrinsic attack behaviour; treat perturbations as
  controlled stimuli, not real-world attack vectors.
- **Imbalance.** Benign flows dominate; several attack types have few samples, which
  is why balancing is train-only and the test set is frozen.

## Why this matters for the XAI / tamper study

SHAP/LIME reveal *which* of these features drove an "attack" verdict — e.g.
`Flow Packets/s` for a DoS flow. The tamper attack's top-1 suppression is precisely
an attempt to make the explanation **stop pointing at that decisive feature** while
the flow is still classified as an attack. The concrete threat: an analyst sees a
flow flagged malicious, but the explanation no longer highlights the rate spike that
actually triggered it — misdirecting incident response.
