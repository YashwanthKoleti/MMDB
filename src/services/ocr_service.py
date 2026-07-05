from PIL import Image
from paddleocr import PaddleOCR

model = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
)

def ocr(image_path:str):
    result = model.predict(image_path)
    text = "\n".join(result[0]["rec_texts"])
    return text