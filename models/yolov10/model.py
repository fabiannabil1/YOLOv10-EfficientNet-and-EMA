from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

from huggingface_hub import PyTorchModelHubMixin
from .card import card_template_text

class YOLOv10(Model, PyTorchModelHubMixin, model_card_template=card_template_text):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False, 
                 names=None):
        super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)
        # Check if model uses EfficientNet and initialize accordingly
        if isinstance(model, str) and 'efficientnet' in model.lower():
            self.init_efficientnet_backbone()

    def init_efficientnet_backbone(self):
        # Contoh: Mengganti backbone model dengan EfficientNet
        from torchvision.models import efficientnet_b7, EfficientNet_B0_Weights
        
        # Memuat model EfficientNet yang sudah pre-trained
        efficientnet = efficientnet_b7(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Mengganti backbone CSPDarknet dengan EfficientNet
        self.model.backbone = efficientnet.features
        
        # Menyesuaikan layer terakhir jika diperlukan
        # Misalnya, jika backbone harus menghasilkan jumlah output tertentu
        num_features = efficientnet.classifier[1].in_features
        self.model.backbone[-1] = nn.Conv2d(num_features, 1024, kernel_size=1)
        
        # Pastikan head tetap kompatibel dengan backbone baru
        self.model.head.input_channels = 1024  # Sesuaikan ini dengan output dari EfficientNet

        # Set model ke mode training atau eval tergantung pada task
        self.model.train() if self.model.training else self.model.eval()


    def push_to_hub(self, repo_name, **kwargs):
        config = kwargs.get('config', {})
        config['names'] = self.names
        config['model'] = self.model.yaml['yaml_file']
        config['task'] = self.task
        kwargs['config'] = config
        super().push_to_hub(repo_name, **kwargs)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
