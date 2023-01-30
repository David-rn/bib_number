import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

class Result:
    def __init__(self) -> None:
        self.bboxes = []
        self.numbers = []
        self.scores = []

    def add_bbox(self, bbox):
        self.bboxes.append(bbox)
    
    def add_number(self, number):
        self.numbers.append(number)
    
    def add_score(self, score):
        self.scores.append(score)
    
    def __getitem__(self, i):
        return self.bboxes[i], self.numbers[i], self.scores[i]


class BibDetector:
    def __init__(self, model_path) -> None:
        print("Loading Bib Detector model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.35
        print("Bib Detector loaded successfully")

    def __call__(self, image):
        im = image[..., ::-1] # OpenCV image (BGR to RGB)
        results = self.model(im, size=640)
        return results.xyxy[0].numpy()


class NumberRecognizer:
    def __init__(self) -> None:
        print("Loading Number Recognizer model...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-str")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-str")
        print("Number Recognizer model loaded")
    
    def __call__(self, im):
        pixel_values = self.processor(im, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


class BibNumberExtractor:
    def __init__(self, bib_det_path) -> None:
        self.bib_detector = BibDetector(bib_det_path)
        self.num_recog = NumberRecognizer()

    @staticmethod
    def crop_number(im):
        pass

    def __call__(self, im):
        det_results = self.bib_detector(im)
        result = Result()
        
        for item in det_results:
            x,y,x_2,y_2 = item[:4]
            bbox = [x,y,x_2-x, y_2-y]
            result.add_bbox(bbox)
            result.add_score(str(item[4]))
            bib = im[int(y):int(y_2), int(x):int(x_2)]
            result.add_number(self.num_recog(bib))

        return result