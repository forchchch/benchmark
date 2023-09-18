import torch
import clip
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform1 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

transform = transforms.Compose(
    [transforms.ToTensor()]
)

# image = preprocess(Image.open("/root/autodl-tmp/benchmark/instance/dog3/01.jpg")).unsqueeze(0).to(device)
# image = preprocess(transform1(Image.open("/root/autodl-tmp/benchmark/instance/dog3/01.jpg"))).unsqueeze(0).to(device)
image = transform1(Image.open("/root/autodl-tmp/benchmark/instance/dog3/01.jpg")).unsqueeze(0).to(device)

print(image.shape)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]