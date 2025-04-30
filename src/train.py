import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from torch.quantization import quantize_dynamic

# Assume CroppedHandwrittenDataset and normalize_arabic_text are already defined

def evaluate_model(model, dataloader, processor, device, max_eval_samples=None):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    num_batches = min(len(dataloader), 50)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss

            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )

            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = batch['text']

            all_predictions.extend(pred_texts)
            all_labels.extend(label_texts)

    avg_loss = total_loss / num_batches

    # Compute CER and WER
    cer = cer_metric.compute(predictions=all_predictions, references=all_labels)
    wer = wer_metric.compute(predictions=all_predictions, references=all_labels)

    return {
        'loss': avg_loss,
        'cer': cer,
        'wer': wer
    }

def train_model(model, train_dataloader, val_dataloader, processor, device,
                num_epochs=30, learning_rate=5e-5, validate_every=2):
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch} - Batch {batch_idx}] Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % validate_every == 0:
            val_metrics = evaluate_model(model, val_dataloader, processor, device)
            print(f"Epoch {epoch} - Validation Loss: {val_metrics['loss']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                model.save_pretrained("/content/drive/MyDrive/trocr_finetuned_best")
                processor.save_pretrained("/content/drive/MyDrive/trocr_finetuned_best")
                print(f"New best model saved with val_loss: {best_val_loss:.4f}")


def quantize_model(model):
    """
    Apply dynamic quantization to speed up inference (mainly for CPU usage)
    """
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def main():
    crops_dir = "/content/drive/MyDrive/handwritten_crops_combined/crops"
    excel_file = "/content/drive/MyDrive/Machathon_Prescription_Digtalization_6.00/Train.xlsx"

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    dataset = CroppedHandwrittenDataset(
        crops_dir=crops_dir,
        excel_file=excel_file,
        processor=processor,
        augment=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optional: Quantize the model to speed up CPU inference
    if device.type == 'cpu':
        model = quantize_model(model)

    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        processor=processor,
        device=device,
        num_epochs=50,
        learning_rate=1e-5
    )


if __name__ == "__main__":
    main()
