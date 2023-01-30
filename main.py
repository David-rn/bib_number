import os
import cv2
import argparse
from bib_number.detector import BibNumberExtractor
from bib_number.utils import draw_bbox

def main(args):
    input_folder = args.input_images
    if not os.path.exists(input_folder):
        raise Exception(f"Input folder '{input_folder}' does not exists")

    # detector = BibDetector('./models/bib.pt')
    model = BibNumberExtractor('./models/bib.pt')
    
    images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for im_path in images:
        im = cv2.imread(os.path.join(input_folder, im_path))
        result = model(im)

        for item in result:
            bbox = item[0]
            number = item[1]
            score = item[2]
            draw_bbox(im, bbox, number)
        
        cv2.imshow("imagen", im)
        cv2.waitKey(0)
  
cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images', type=str, required=True, help='Path to images')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)