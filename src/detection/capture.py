#!/usr/bin/env python3

import cv2
import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms


CLASSES =["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
          "del", "nothing", "space"]


def load_model(path:str) -> nn.Module:
    return torch.jit.load(path)





def main():
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--trained_model_path",
                        default="src/model/asl_baseline.pt")
    
    args = parser.parse_args()
    
    model = load_model(args.trained_model_path)

    cap = cv2.VideoCapture(0)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while True:
        
        ret, frame = cap.read()
        input_tensor = preprocess(frame).unsqueeze(0)
        
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            
        predicted_class = torch.argmax(output).item()
        
        prediction = CLASSES[predicted_class]
        print(f"Predicted Class: {prediction}")
        cv2.putText(frame, f"Class: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()