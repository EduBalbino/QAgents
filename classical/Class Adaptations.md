# Class Adaptations

This file records how attack classes were adapted from the original trio datasets into the final training taxonomy.

## Source Datasets

Merged from subfolders under `data/`:

- `CIC-BCCC-NRC-Edge-IIoTSet-2022`
- `CIC-BCCC-NRC-IoT-2023-Original Training and Testing`
- `CIC-BCCC-NRC-UQ-IOT-2022`

Pipeline scripts used:

- `classical/prepare_specialist_dataset.py` (Polars merge + label remap, replaces old 3-script pipeline)

## Original Label Space (Before Final Remap)

From `data/processed/trio_multiclass_from_subfolders.csv`:

- Total rows: `14,860,312`
- Unique original attack labels: `22`

Original labels with provenance (`label -> source dataset (rows)`):

1. `ACK Flood` -> `CIC-BCCC-NRC-UQ-IOT-2022` (`2,905,508`)
2. `Backdoor` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`900`)
3. `Benign Traffic` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`2,154,864`)
4. `DDoS ACK Fragmentation` -> `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` (`405,824`)
5. `DDoS HTTP Flood` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`15,795`)
6. `DDoS ICMP Fragmentation` -> `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` (`9,775`)
7. `DDoS PSHACK Flood` -> `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` (`2,547,096`)
8. `DDoS RSTFIN Flood` -> `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` (`3,938,280`)
9. `DDoS TCP SYN Flood` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`843,527`)
10. `MITM ARP Spoofing` -> `CIC-BCCC-NRC-UQ-IOT-2022` (`320`)
11. `Mirai UDP Plain` -> `CIC-BCCC-NRC-IoT-2023-Original Training and Testing` (`1,630`)
12. `OS Fingerprinting` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`135`)
13. `Password Attack` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`179,170`)
14. `Port Scanning` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`9,987`)
15. `Ransomware` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`639`)
16. `Recon Port Scan` -> `CIC-BCCC-NRC-UQ-IOT-2022` (`16,139`)
17. `SQL Injection` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`8,755`)
18. `SYN Flood` -> `CIC-BCCC-NRC-UQ-IOT-2022` (`1,803,982`)
19. `Telnet Brute Force` -> `CIC-BCCC-NRC-UQ-IOT-2022` (`694`)
20. `Uploading Attack` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`9,516`)
21. `Vulnerability Scanner` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`5,424`)
22. `XSS` -> `CIC-BCCC-NRC-Edge-IIoTSet-2022` (`2,352`)

## RF Alignment Stage

`rf_attack_label_alignment.py` was used to estimate cross-dataset label similarity from feature behavior (Random Forest, bidirectional probability matching, symmetric score).

Notes:

- This stage helps identify likely correspondences.
- Due domain shift, RF matches can include noisy pairings.
- Final remap was constrained to a practical, stable taxonomy for training.

## Final Taxonomy Target (Specialist-Agent Aligned)

Current final policy:

- Benign rows must map to `NORMAL`.
- Attack rows must map to the specialist attack classes:
  - `Backdoor`
  - `DDoS_HTTP`
  - `DDoS_ICMP`
  - `DDoS_TCP`
  - `DDoS_UDP`
  - `Fingerprinting`
  - `MITM`
  - `Password`
  - `Port_Scanning`
  - `Ransomware`
  - `SQL_injection`
  - `Uploading`
  - `Vulnerability_scanner`
  - `XSS`
- Any unmapped/unmentioned attack label should go to `Others`.

Single final artifact:

- `data/processed/trio_multiclass_final_single.csv`

Required output columns in this file:

- `Attack_label` (multiclass final label, includes `NORMAL`)
- `Attack_type` (binary: `0` benign / `1` attack)

## Column Naming Convention

Feature columns are normalized to **lowercase.dot** notation (CIC-FlowMeter flow aggregates).
Where a semantic match exists to the CIC-IoT-Dataset-2023 Wireshark schema, the target name is used.

Semantic overrides applied:

| CIC-FlowMeter original | Normalized name | Reason |
|------------------------|-----------------|--------|
| `Timestamp` | `frame.time` | Matches Wireshark schema |
| `Src IP` | `ip.src_host` | Matches Wireshark schema |
| `Dst IP` | `ip.dst_host` | Matches Wireshark schema |
| `Src Port` | `tcp.srcport` | Matches Wireshark schema |
| `Dst Port` | `tcp.dstport` | Matches Wireshark schema |
| `Label` | `Attack_label` | Matches Wireshark schema |

All other feature columns: `lower().replace(' ','.').replace('/','.'))`, e.g.:

- `Flow Duration` -> `flow.duration`
- `Fwd Packet Length Max` -> `fwd.packet.length.max`
- `Flow Bytes/s` -> `flow.bytes.s`
- `FWD Init Win Bytes` -> `fwd.init.win.bytes`
- `Down/Up Ratio` -> `down.up.ratio`

Full column list (85 columns): `flow.id`, `ip.src_host`, `tcp.srcport`, `ip.dst_host`, `tcp.dstport`, `protocol`, `frame.time`, `flow.duration`, `total.fwd.packet`, `total.bwd.packets`, `total.length.of.fwd.packet`, `total.length.of.bwd.packet`, `fwd.packet.length.max`, `fwd.packet.length.min`, `fwd.packet.length.mean`, `fwd.packet.length.std`, `bwd.packet.length.max`, `bwd.packet.length.min`, `bwd.packet.length.mean`, `bwd.packet.length.std`, `flow.bytes.s`, `flow.packets.s`, `flow.iat.mean`, `flow.iat.std`, `flow.iat.max`, `flow.iat.min`, `fwd.iat.total`, `fwd.iat.mean`, `fwd.iat.std`, `fwd.iat.max`, `fwd.iat.min`, `bwd.iat.total`, `bwd.iat.mean`, `bwd.iat.std`, `bwd.iat.max`, `bwd.iat.min`, `fwd.psh.flags`, `bwd.psh.flags`, `fwd.urg.flags`, `bwd.urg.flags`, `fwd.header.length`, `bwd.header.length`, `fwd.packets.s`, `bwd.packets.s`, `packet.length.min`, `packet.length.max`, `packet.length.mean`, `packet.length.std`, `packet.length.variance`, `fin.flag.count`, `syn.flag.count`, `rst.flag.count`, `psh.flag.count`, `ack.flag.count`, `urg.flag.count`, `cwr.flag.count`, `ece.flag.count`, `down.up.ratio`, `average.packet.size`, `fwd.segment.size.avg`, `bwd.segment.size.avg`, `fwd.bytes.bulk.avg`, `fwd.packet.bulk.avg`, `fwd.bulk.rate.avg`, `bwd.bytes.bulk.avg`, `bwd.packet.bulk.avg`, `bwd.bulk.rate.avg`, `subflow.fwd.packets`, `subflow.fwd.bytes`, `subflow.bwd.packets`, `subflow.bwd.bytes`, `fwd.init.win.bytes`, `bwd.init.win.bytes`, `fwd.act.data.pkts`, `fwd.seg.size.min`, `active.mean`, `active.std`, `active.max`, `active.min`, `idle.mean`, `idle.std`, `idle.max`, `idle.min`, `Attack_label`, `Attack_type`

## Final Mapping Used

Benign handling:

- `Benign Traffic` -> `NORMAL`

Attack handling:

- `DDoS HTTP Flood` -> `DDoS_HTTP`
- `DDoS ICMP Fragmentation` -> `DDoS_ICMP`
- `DDoS TCP SYN Flood` -> `DDoS_TCP`
- `DDoS ACK Fragmentation` -> `DDoS_TCP`
- `Mirai UDP Plain` -> `DDoS_UDP`
- `Backdoor` -> `Backdoor`
- `OS Fingerprinting` -> `Fingerprinting`
- `MITM ARP Spoofing` -> `MITM`
- `Password Attack` -> `Password`
- `Telnet Brute Force` -> `Password`
- `Port Scanning` -> `Port_Scanning`
- `Recon Port Scan` -> `Port_Scanning`
- `Ransomware` -> `Ransomware`
- `SQL Injection` -> `SQL_injection`
- `Uploading Attack` -> `Uploading`
- `Vulnerability Scanner` -> `Vulnerability_scanner`
- `XSS` -> `XSS`

Alias merges explicitly applied:

- `DDoS ACK Fragmentation` merged into `DDoS_TCP`
- `Telnet Brute Force` merged into `Password`
- `Recon Port Scan` merged into `Port_Scanning`

Additional remaps applied for deployment consistency:

- `ACK Flood` -> `DDoS_TCP`
- `SYN Flood` -> `DDoS_TCP`
- `DDoS PSHACK Flood` -> `DDoS_TCP`
- `DDoS RSTFIN Flood` -> `DDoS_TCP`
