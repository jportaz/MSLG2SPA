import pandas as pd
import sacrebleu

def compute_bleu(path, sep=None):
    """
    Reads a file with columns:
      MSL | SPA | MSL-predicted
    and computes corpus BLEU:
      hypothesis = MSL-predicted
      reference  = MSL
    """

    # Try to infer separator if not provided
    if sep is None:
        if path.endswith(".tsv"):
            sep = "\t"
        else:
            sep = ","

    df = pd.read_csv(path, sep=sep)

    #required = ["MSL", "SPA", "MSL-predicted"]
    #missing = [c for c in required if c not in df.columns]
    #if missing:
    #    raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Convert to strings and drop empty rows
    refs = df["MSLG"].fillna("").astype(str).tolist()
    hyps = df["SPA"].fillna("").astype(str).tolist()

    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    refs = [r for r, h in valid]
    hyps = [h for r, h in valid]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    print(f"BLEU score: {bleu.score:.4f}")
    print(bleu.format())

    return bleu

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/dev/stdin")
    args = parser.parse_args()

    compute_bleu(args.file, sep="\t")