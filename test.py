import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

try:
	import gradio as gr
except Exception:
	gr = None


def load_model(weights_path="twitter_classifier.pth", device=None):
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	weights = models.ResNet152_Weights.IMAGENET1K_V2
	model = models.resnet152(weights=weights)
	
	# determine number of classes from data/
	try:
		dataset = ImageFolder(root="data/")
		num_classes = len(dataset.classes)
		class_names = dataset.classes
	except Exception:
		# fallback if data folder not present
		num_classes = 2
		class_names = [f"class_{i}" for i in range(num_classes)]

	model.fc = nn.Linear(model.fc.in_features, num_classes)
	
	# wrap model (must match training setup)
	class ResNet152WithDropout(nn.Module):
		def __init__(self, base_model, num_classes):
			super().__init__()
			self.base = base_model
			
		def forward(self, x):
			return self.base(x)
	
	wrapped_model = ResNet152WithDropout(model, num_classes)
	wrapped_model.to(device)

	if os.path.exists(weights_path):
		state = torch.load(weights_path, map_location=device)
		try:
			wrapped_model.load_state_dict(state)
		except Exception:
			# allow loading partial/state-dict variations
			wrapped_model.load_state_dict(state, strict=False)
	else:
		print(f"Warning: weights file '{weights_path}' not found. Using randomly initialized model.")

	wrapped_model.eval()
	return wrapped_model, class_names, device


_PREPROCESS = transforms.Compose([
	transforms.Resize((384, 384)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_prediction_image(pil_image, model, class_names, device):
	# ensure RGB
	img = pil_image.convert('RGB')
	input_tensor = _PREPROCESS(img).unsqueeze(0).to(device)

	with torch.no_grad():
		outputs = model(input_tensor)
		probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

	# matplotlib bar chart: categories as columns
	fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(class_names)), 4))
	x = np.arange(len(class_names))
	bars = ax.bar(x, probs, color='tab:blue')
	ax.set_xticks(x)
	ax.set_xticklabels(class_names, rotation=45, ha='right')
	ax.set_ylim(0, 1)
	ax.set_ylabel('Probability')
	ax.set_title('Class probabilities')

	# label bars with percentages
	for bar, p in zip(bars, probs):
		ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f"{p:.2f}", ha='center', va='bottom', fontsize=8)

	plt.tight_layout()

	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	plt.close(fig)
	buf.seek(0)
	out_img = Image.open(buf).convert('RGB')
	return out_img


def build_app(weights_path="twitter_classifier.pth"):
	model, class_names, device = load_model(weights_path=weights_path)

	def predict(image: Image.Image):
		if image is None:
			return None
		try:
			viz = make_prediction_image(image, model, class_names, device)
			return viz
		except Exception as e:
			print(f"Prediction error: {e}")
			# return a small error image
			fig, ax = plt.subplots(figsize=(6, 2))
			ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
			ax.axis('off')
			buf = io.BytesIO()
			fig.savefig(buf, format='png')
			plt.close(fig)
			buf.seek(0)
			return Image.open(buf).convert('RGB')

	if gr is None:
		print("Gradio is not installed. Install it with `pip install gradio` and re-run this script.")
		return None

	iface = gr.Interface(fn=predict,
						 inputs=gr.Image(type='pil', label='Input image'),
						 outputs=gr.Image(type='pil', label='Prediction visualization'),
						 title='Twitter image classifier',
						 description='Uploads an image and shows class probabilities as columns (matplotlib)')
	return iface


if __name__ == '__main__':
	iface = build_app()
	if iface is not None:
		print('Launching Gradio app...')
		iface.launch(server_name='0.0.0.0')

