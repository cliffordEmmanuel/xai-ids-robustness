# Data Dictionary — CICIDS2017 & UNSW-NB15

Field descriptions for the two benchmark datasets used in this project.

**Sources.** CICIDS2017 has no official CIC data dictionary; its columns are the
output of [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter) and are
described here from the tool's documentation and Sharafaldin et al. (2018).
UNSW-NB15 columns are from the official `UNSW-NB15_features.csv` and Moustafa &
Slay (2015).

> **Verified** against the official `NUSW-NB15_features.csv` (the 49 raw
> features). Note that the training/testing **partition** used by the loader
> differs from that raw file: it drops `srcip`/`sport`/`dstip`/`dsport` and the
> `Stime`/`Ltime` timestamps, renames some columns to lowercase
> (`smeansz`→`smean`, `dmeansz`→`dmean`, `res_bdy_len`→`response_body_len`,
> `Sintpkt`→`sinpkt`, …), and adds `id` and the derived `rate`. The table below
> uses the **partition** names (what the loader sees).

Both datasets are binary-labelled in this project (BENIGN/normal → 0, any attack
→ 1). Identifier columns are dropped by the loaders (see notes per dataset).

---

## 1. CICIDS2017 (CICFlowMeter features)

Each row is one bidirectional network **flow**. "Fwd" = forward direction
(source→destination), "Bwd" = backward (destination→source). Many features come
in Max/Min/Mean/Std families over packet sizes or inter-arrival times (IAT).

**Dropped by the loader (identity/leakage):** `Flow ID`, source/destination IP,
source port, `Destination Port` (by default), `Timestamp`, and the duplicated
`Fwd Header Length.1`.

### Duration & counts

| Feature | Meaning |
|---|---|
| Flow Duration | Total flow duration (microseconds) |
| Total Fwd Packets | Number of packets in the forward direction |
| Total Backward Packets | Number of packets in the backward direction |
| Total Length of Fwd Packets | Total bytes sent forward |
| Total Length of Bwd Packets | Total bytes sent backward |

### Packet length (bytes per packet)

| Feature | Meaning |
|---|---|
| Fwd Packet Length Max / Min / Mean / Std | Stats of forward packet sizes |
| Bwd Packet Length Max / Min / Mean / Std | Stats of backward packet sizes |
| Min / Max Packet Length | Smallest / largest packet in the flow |
| Packet Length Mean / Std / Variance | Stats over all packets |
| Average Packet Size | Mean size of a packet in the flow |
| Avg Fwd Segment Size | Mean segment size forward |
| Avg Bwd Segment Size | Mean segment size backward |

### Rates

| Feature | Meaning |
|---|---|
| Flow Bytes/s | Bytes per second over the flow (**can be NaN/Inf**) |
| Flow Packets/s | Packets per second over the flow (**can be NaN/Inf**) |
| Fwd Packets/s | Forward packets per second |
| Bwd Packets/s | Backward packets per second |
| Down/Up Ratio | Ratio of download to upload |

### Inter-arrival time (IAT, time between packets)

| Feature | Meaning |
|---|---|
| Flow IAT Mean / Std / Max / Min | Time between packets over the whole flow |
| Fwd IAT Total / Mean / Std / Max / Min | Time between forward packets |
| Bwd IAT Total / Mean / Std / Max / Min | Time between backward packets |

### TCP flags

| Feature | Meaning |
|---|---|
| Fwd / Bwd PSH Flags | Count of PSH flags per direction |
| Fwd / Bwd URG Flags | Count of URG flags per direction |
| FIN / SYN / RST / PSH / ACK / URG / CWE / ECE Flag Count | Count of each TCP flag in the flow |

### Headers & window

| Feature | Meaning |
|---|---|
| Fwd Header Length | Total bytes used for forward headers |
| Bwd Header Length | Total bytes used for backward headers |
| Init_Win_bytes_forward | Initial TCP window (forward) |
| Init_Win_bytes_backward | Initial TCP window (backward) |
| act_data_pkt_fwd | Forward packets with ≥1 byte of TCP payload |
| min_seg_size_forward | Minimum segment size observed forward |

### Bulk transfer

| Feature | Meaning |
|---|---|
| Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate | Forward bulk-transfer averages |
| Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate | Backward bulk-transfer averages |

### Subflows

| Feature | Meaning |
|---|---|
| Subflow Fwd Packets / Bytes | Average packets / bytes in a forward subflow |
| Subflow Bwd Packets / Bytes | Average packets / bytes in a backward subflow |

### Active / idle periods

| Feature | Meaning |
|---|---|
| Active Mean / Std / Max / Min | Time a flow was active before going idle |
| Idle Mean / Std / Max / Min | Time a flow was idle before becoming active |

### Label

| Feature | Meaning |
|---|---|
| Label | `BENIGN` or an attack name (DoS Hulk, PortScan, DDoS, …) → binarised to 0/1 |

---

## 2. UNSW-NB15 (training/testing partition columns)

The loader uses the pre-partitioned `UNSW_NB15_{training,testing}-set.csv`
(≈45 columns). Types: **cat** = categorical (one-hot encoded by the loader),
**int/float** = numeric, **bin** = binary flag.

**Dropped by the loader:** `id` (row index) and `attack_cat` (kept only in meta).

### Flow / basic

| Feature | Type | Meaning |
|---|---|---|
| dur | float | Total flow duration (seconds) |
| proto | cat | Transaction protocol (tcp, udp, …) |
| service | cat | Application service (http, ftp, dns, smtp; `-` if none) |
| state | cat | Connection state (ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, `-`) |
| spkts | int | Source → destination packet count |
| dpkts | int | Destination → source packet count |
| sbytes | int | Source → destination bytes |
| dbytes | int | Destination → source bytes |
| rate | float | Packets per second (derived; partition-only, not in the raw 49) |
| sttl | int | Source → destination time-to-live |
| dttl | int | Destination → source time-to-live |
| sload | float | Source bits per second |
| dload | float | Destination bits per second |
| sloss | int | Source packets retransmitted / dropped |
| dloss | int | Destination packets retransmitted / dropped |

### Timing

| Feature | Type | Meaning |
|---|---|---|
| sinpkt | float | Source inter-packet arrival time (ms) |
| dinpkt | float | Destination inter-packet arrival time (ms) |
| sjit | float | Source jitter (ms) |
| djit | float | Destination jitter (ms) |
| tcprtt | float | TCP setup round-trip time (synack + ackdat) |
| synack | float | Time between SYN and SYN-ACK |
| ackdat | float | Time between SYN-ACK and ACK |

### TCP

| Feature | Type | Meaning |
|---|---|---|
| swin | int | Source TCP window advertisement |
| dwin | int | Destination TCP window advertisement |
| stcpb | int | Source TCP base sequence number |
| dtcpb | int | Destination TCP base sequence number |
| smean | int | Mean packet size transmitted by source |
| dmean | int | Mean packet size transmitted by destination |

### HTTP / content

| Feature | Type | Meaning |
|---|---|---|
| trans_depth | int | Pipelined depth of the HTTP request/response |
| response_body_len | int | Uncompressed size of server HTTP content |

### Connection counters (last 100 connections)

| Feature | Type | Meaning |
|---|---|---|
| ct_srv_src | int | Same service and source address |
| ct_srv_dst | int | Same service and destination address |
| ct_dst_ltm | int | Same destination address |
| ct_src_ltm | int | Same source address |
| ct_src_dport_ltm | int | Same source address and destination port |
| ct_dst_sport_ltm | int | Same destination address and source port |
| ct_dst_src_ltm | int | Same source and destination address |
| ct_state_ttl | int | Count per state for a given source/dest TTL range |
| ct_flw_http_mthd | int | Flows using HTTP GET/POST methods |
| ct_ftp_cmd | int | Flows with a command in an FTP session |

### Flags

| Feature | Type | Meaning |
|---|---|---|
| is_ftp_login | bin | 1 if the FTP session was authenticated |
| is_sm_ips_ports | bin | 1 if source and destination IP and ports are equal |

### Labels

| Feature | Type | Meaning |
|---|---|---|
| attack_cat | cat | Attack family (Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) — dropped, kept in meta |
| label | bin | 0 = normal, 1 = attack (the target) |

---

## Notes for this project

- Both datasets are **binarised** (attack vs normal) and their identifier columns
  dropped, so the features above that survive into the model are the flow
  statistics — the space in which SHAP/LIME attributions and the tamper attack
  operate.
- UNSW's categoricals (`proto`, `service`, `state`) become one-hot columns, so
  some attributions land on binary dummy features; the tamper target rule accounts
  for this.
- `Flow Bytes/s` and `Flow Packets/s` (CICIDS2017) carry NaN/Inf and are cleaned
  during loading.
