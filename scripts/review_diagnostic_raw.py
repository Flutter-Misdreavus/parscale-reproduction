#!/usr/bin/env python3
"""CPU-only independent reconstruction from remote diagnostic arrays and tokens.

Does not import evaluator/diagnostic code or perform model inference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def close(a, b, label):
    if not math.isfinite(float(a)) or not math.isclose(float(a),float(b),rel_tol=1e-5,abs_tol=2e-6):
        raise AssertionError((label,float(a),float(b)))


def target_lp(logits, tokens):
    values=[]
    for start in range(0,len(tokens)-1,32):
        stop=min(start+32,len(tokens)-1)
        x=np.asarray(logits[start:stop],dtype=np.float64)
        maximum=x.max(axis=1)
        normalizer=maximum+np.log(np.exp(x-maximum[:,None]).sum(axis=1))
        values.extend((x[np.arange(stop-start),tokens[start+1:stop+1]]-normalizer).tolist())
    return np.asarray(values)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=args.run_dir.resolve()
    locks=json.loads((root/"locks.json").read_text())
    d=json.loads((run/"diagnostics.json").read_text())
    assert d["status"]=="complete" and set(d["models"])=={"P1","P2","P4"}
    verified=[]
    for item in d["raw_manifest"]:
        p=(run/item["path"]).resolve()
        assert p.is_relative_to(run)
        assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
        verified.append(item["path"])

    # Reconstruct the exact validation text/targets, independently of worker helpers.
    from datasets import load_dataset
    from transformers import AutoTokenizer
    spec=locks["dataset"]
    data=load_dataset(spec["id"],spec["subset"],revision=spec["revision"],
                      split="validation",cache_dir=str(Path(os.environ["HF_HOME"])/"datasets"))
    text="\n\n".join(data["text"])
    paths=json.loads((run/"raw/downloads/model_paths.json").read_text())
    tok=AutoTokenizer.from_pretrained(paths["1"],local_files_only=True,trust_remote_code=False)
    ids=np.asarray(tok(text,add_special_tokens=False)["input_ids"],dtype="<i4")
    manifest=json.loads((run/"data-manifest.json").read_text())
    assert hashlib.sha256(text.encode()).hexdigest()==manifest["text_sha256"]
    assert hashlib.sha256(ids.tobytes()).hexdigest()==manifest["token_ids_sha256"]
    result={"run_id":run.name,"all_raw_hashes_verified":True,"raw_files_verified":verified,
            "validation_text_and_token_hashes_reconstructed":True,"models":{},
            "review_code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
            "model_inference_rerun":False,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    for key, model in sorted(d["models"].items()):
        folder=run/"raw/diagnostics"/key
        totals={"float32":0.0,"bfloat16":0.0}
        max32=max16=0.0
        for bi in range(2):
            tokens=ids[bi*1024:(bi+1)*1024]
            ref=np.load(folder/f"P_float32_loss_b{bi}.npy",mmap_mode="r")
            lp=target_lp(ref,tokens)
            record=model["loss_1024"]["fp32"][bi]
            close(-lp.sum()/1023,record["nll_nats_per_token"],f"{key} FP32 raw NLL")
            assert np.max(np.abs(lp-np.asarray(record["target_logprobs"])))<1e-5
            totals["float32"]-=float(lp.sum())
            bf=model["loss_1024"]["bfloat16"][bi]
            totals["bfloat16"]-=sum(bf["target_logprobs"])
            close(-sum(bf["target_logprobs"])/1023,bf["nll_nats_per_token"],f"{key} BF16 saved LP")
        sequences=[]
        for sid in range(2):
            with np.load(folder/f"P_float32_seq{sid}.npz") as fp, np.load(folder/f"P_bfloat16_seq{sid}.npz") as bf:
                tokens=fp["tokens"]
                assert np.array_equal(tokens,ids[sid*1024:sid*1024+64])
                assert np.array_equal(tokens,bf["tokens"])
                for dtype,arrays in [("float32",fp),("bfloat16",bf)]:
                    record=model["cache_sequences"][dtype][str(sid)]
                    assert hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest()==record["sequence"]["token_ids_sha256_i32le"]
                    for prefill in (1,32):
                        error=np.abs(arrays[f"cached_p{prefill}"]-arrays["full"])
                        actual=float(error.max())
                        close(actual,record[f"full_vs_cached_p{prefill}"]["max_abs"],f"{key} raw cache error")
                        if dtype=="float32":max32=max(max32,actual)
                        else:max16=max(max16,actual)
                for path in ("full","cached_p1","cached_p32"):
                    error=np.abs(bf[path]-fp[path])
                    rec=model["cache_sequences"]["bfloat16"][str(sid)][path+"_vs_fp32"]
                    close(float(error.max()),rec["max_abs"],f"{key} {path} cross precision")
                    observed=np.abs(target_lp(bf[path],tokens)-target_lp(fp[path],tokens))
                    close(float(observed.max()),rec["target_logprob_abs_diff_max"],f"{key} target LP")
                sequences.append({"sequence":sid,"token_count":len(tokens),"raw_comparisons_verified":True})
        result["models"][key]={"tokens_scored":2046,
            "fp32_nll_reconstructed_from_raw_logits":totals["float32"]/2046,
            "bf16_nll_recomputed_from_saved_target_logprobs":totals["bfloat16"]/2046,
            "max_fp32_cache_error":max32,"max_bf16_cache_error":max16,
            "sequences":sequences}
    timing=json.loads((run/"timing.json").read_text())
    result["timing_recomputed"]={}
    for key,r in timing["models"].items():
        assert len(r["raw_seconds"])==10 and r["output_lengths"]==[160]*13
        median=float(np.median(r["raw_seconds"]));p90=float(np.percentile(r["raw_seconds"],90))
        close(median,r["median_seconds"],key+" timing median")
        close(p90,r["p90_seconds"],key+" timing p90")
        result["timing_recomputed"][key]={"median_seconds":median,"p90_seconds":p90}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
