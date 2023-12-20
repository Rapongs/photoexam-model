from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.cloud.vision_v1 import types
import tensorflow as tf
import transformers
import numpy as np
import requests
from pydantic import BaseModel

app = FastAPI()

# Load BERT model
MAXLEN = 256
BATCH_SIZE = 48
labels = ["contradiction", "entailment", "neutral"]

class BertSemanticDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "indolem/indobert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=MAXLEN,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

def check_similarity(sentence1, sentence2):
    model = tf.keras.models.load_model('iter2FT.h5', custom_objects={"TFBertModel": transformers.TFBertModel})
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    pred = labels[idx]
    if pred == "entailment" and proba[idx] >= 0.60:
        return 1
    elif pred == "neutral" and proba[idx] >= 0.70:
        return 1
    else:
        return 0

# Google Cloud Vision API
client = vision.ImageAnnotatorClient()

def perform_ocr(image_content):
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)
    full_txt = response.full_text_annotation.text
    encoded_text = full_txt.encode('ascii', 'ignore')
    decoded_text = encoded_text.decode()
    return decoded_text

class ImageProcessRequest(BaseModel):
    image_url: str
    key_answer: str

@app.post("/process_image")
async def process_image(request_data: ImageProcessRequest):
    try:
        image_url = request_data.image_url
        key_answer = request_data.key_answer
        
        response = requests.get(image_url)
        response.raise_for_status()  # Raise HTTPError for bad responses

        contents = response.content
        # Perform OCR
        ocr_text = perform_ocr(contents)

        # Check similarity
        similarity_result = check_similarity(ocr_text, key_answer)
        return JSONResponse(content={"OCRText": ocr_text, "SimilarityResult": similarity_result})
    except requests.RequestException as e:
        # Handle request-related exceptions
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")