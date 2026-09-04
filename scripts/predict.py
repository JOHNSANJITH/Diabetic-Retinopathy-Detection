import argparse
from drd.inference import load_model, predict_image
from drd.explainability import gradcam

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--image",required=True); p.add_argument("--gradcam-output"); args=p.parse_args()
    model=load_model(args.model); print(predict_image(model,args.image))
    if args.gradcam_output: print("Grad-CAM:",gradcam(model,args.image,args.gradcam_output)[1:])

if __name__ == "__main__": main()
