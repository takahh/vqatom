
#!/usr/bin/env python3
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path("/Users/taka/Downloads")
INPUT = BASE / (
    "41589_2011_BFnchembio530_MOESM137_ESM_"
    "Table_S1__Kinome_pKi_Data_with_.csv"
)
OUT = BASE / "metz_prepared"
OUT.mkdir(parents=True, exist_ok=True)

# Legacy names and potentially ambiguous target names.
# These must be manually verified, even if a candidate is found.
LEGACY_NAMES = {
    "CDC2": "CDK1",
    "CDC2L6": "CDK11A",
    "STK6": "AURKA",
    "STK12": "AURKB",
    "TAO1": "TAOK1",
    "TOPK": "PBK",
    "ACK1": "TNK2",
    "DCAMKL1": "DCLK1",
    "GPRK5": "GRK5",
    "SGK": "SGK1",
    "KIAA1811": "TSSK4",
    "STK22B": "TSSK1B",
    "STK22D": "TSSK2",
    "PRKCN": "PRKD3",
    "MINK": "MINK1",
}

session = requests.Session()
session.headers.update({
    "User-Agent": "MetzDTIResearch/1.0",
    "Accept": "application/json",
})


def search_uniprot(gene):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": (
            f"gene_exact:{gene} "
            "AND organism_id:9606 AND reviewed:true"
        ),
        "format": "json",
        "size": 20,
    }

    for attempt in range(4):
        try:
            response = session.get(
                url, params=params, timeout=40
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except (requests.RequestException, ValueError) as e:
            print(f"  Retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)

    return None


def get_gene_names(entry):
    names = set()
    for gene in entry.get("genes", []):
        primary = gene.get("geneName", {}).get("value")
        if primary:
            names.add(primary.upper())

        for key in ("synonyms", "orderedLocusNames"):
            for item in gene.get(key, []):
                value = item.get("value")
                if value:
                    names.add(value.upper())
    return names


def resolve_target(name):
    is_legacy = name in LEGACY_NAMES
    query_name = LEGACY_NAMES.get(name, name)

    results = search_uniprot(query_name)

    row = {
        "target": name,
        "query_gene": query_name,
        "uniprot_id": "",
        "protein_sequence": "",
        "sequence_length": 0,
        "candidate_count": 0,
        "status": "review",
        "reason": "",
    }

    if results is None:
        row["reason"] = "API_ERROR"
        return row

    # Require the current official gene name to match exactly.
    exact = []
    for entry in results:
        genes = entry.get("genes", [])
        if not genes:
            continue

        primary = genes[0].get(
            "geneName", {}
        ).get("value", "")

        if primary.upper() == query_name.upper():
            exact.append(entry)

    row["candidate_count"] = len(exact)

    if len(exact) != 1:
        row["reason"] = (
            "NO_EXACT_MATCH" if not exact
            else "MULTIPLE_MATCHES"
        )
        return row

    entry = exact[0]
    accession = entry.get("primaryAccession", "")
    sequence = entry.get("sequence", {}).get("value", "")

    row["uniprot_id"] = accession
    row["protein_sequence"] = sequence
    row["sequence_length"] = len(sequence)

    if not sequence:
        row["reason"] = "NO_SEQUENCE"
    elif is_legacy:
        row["reason"] = "LEGACY_NAME_VERIFY"
    else:
        row["status"] = "accepted"
        row["reason"] = "EXACT_GENE_MATCH"

    return row


def main():
    df = pd.read_csv(INPUT, low_memory=False)
    print("Input shape:", df.shape)

    # Actual Metz Table S1 layout
    targets = list(df.columns[14:])
    print("Targets:", len(targets))

    # Only retain compounds with SMILES.
    smiles = df["Canonical_Smiles"].astype("string")
    mask = smiles.notna() & smiles.str.strip().ne("")
    df = df.loc[mask].copy()

    print("Rows with SMILES:", len(df))

    # Convert affinity matrix to long format.
    long = df.melt(
        id_vars=["Cmpd_ID", "Canonical_Smiles"],
        value_vars=targets,
        var_name="target",
        value_name="pKi_raw",
    )

    # Only strictly numeric measurements.
    # Censored values such as <5.4 are excluded.
    long["pKi"] = pd.to_numeric(
        long["pKi_raw"], errors="coerce"
    )
    long = long.dropna(subset=["pKi"]).copy()

    print("Numeric pKi pairs:", len(long))

    # UniProt mapping, with resumable checkpoint.
    mapping_path = OUT / "metz_target_mapping.csv"

    if mapping_path.exists():
        old = pd.read_csv(mapping_path).fillna("")
        records = {
            str(row["target"]): row.to_dict()
            for _, row in old.iterrows()
        }
    else:
        records = {}

    for i, target in enumerate(targets, 1):
        if target in records:
            continue

        print(f"[{i}/{len(targets)}] {target}")
        records[target] = resolve_target(target)

        pd.DataFrame(
            list(records.values())
        ).to_csv(mapping_path, index=False)

        time.sleep(0.2)

    mapping = pd.DataFrame([
        records[target] for target in targets
    ])
    mapping.to_csv(mapping_path, index=False)

    review = mapping[
        mapping["status"] != "accepted"
    ].copy()
    review.to_csv(
        OUT / "metz_targets_need_review.csv",
        index=False,
    )

    accepted = mapping[
        mapping["status"] == "accepted"
    ][["target", "uniprot_id", "protein_sequence"]]

    dti = long.merge(
        accepted,
        on="target",
        how="inner",
        validate="many_to_one",
    )

    dti = dti.rename(columns={
        "Canonical_Smiles": "SMILES",
    })

    dti = dti[[
        "Cmpd_ID",
        "SMILES",
        "target",
        "uniprot_id",
        "protein_sequence",
        "pKi",
    ]]

    # Preserve measurements, but flag duplicate entity pairs.
    dti["duplicate_pair"] = dti.duplicated(
        subset=["SMILES", "uniprot_id"],
        keep=False,
    )

    dti.to_csv(
        OUT / "metz_dti_candidate.csv",
        index=False,
    )

    print("\n=== Summary ===")
    print("Targets:", len(targets))
    print("Accepted:", len(accepted))
    print("Need review:", len(review))
    print("DTI candidate pairs:", len(dti))
    print("Duplicate pair rows:", dti["duplicate_pair"].sum())
    print("Output:", OUT)


if __name__ == "__main__":
    main()
