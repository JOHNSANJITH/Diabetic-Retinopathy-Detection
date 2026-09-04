import argparse
from pathlib import Path
import numpy as np
from drd.inference import load_model
from drd.metrics import evaluate_predictions, save_metrics
from drd.data import _dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--output-dir", default="artifacts/evaluation"); p.add_argument("--batch-size", type=int, default=16); args=p.parse_args()
    model=load_model(args.model); ds=_dataset(args.data_dir, (299,299), args.batch_size, False, 42)
    ys=[]; ps=[]
    for x,y in ds: ys.extend(np.argmax(y.numpy(),axis=1)); ps.extend(model.predict(x,verbose=0))
    metrics=evaluate_predictions(np.array(ys),np.array(ps)); save_metrics(metrics,Path(args.output_dir)/"metrics.json"); print(metrics)

if __name__ == "__main__": main()
