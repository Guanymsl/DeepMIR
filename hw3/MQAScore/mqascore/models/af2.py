import json
import sys
import torch
import yaml

from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .base import LALMBackend
from ..scoring import yes_no_ids, pyes_from_logits

class AudioFlamingo2Backend(LALMBackend):
    name = 'af2'
    model_id = 'nvidia/audio-flamingo-2-0.5B'
    default_fix_seconds = 10

    def __init__(self, model_id=None):
        self.model_id = model_id or self.model_id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = (
            torch.bfloat16 if self.device.type == 'cuda'
            else torch.float32
        )

        root = Path(__file__).resolve().parents[3]
        af_dir = root / 'AudioFlamingo2' / 'inference_HF_pretrained'

        if not af_dir.exists():
            raise FileNotFoundError(f'AF2 source not found: {af_dir}')

        sys.path.insert(0, str(af_dir))
        
        from src.factory import create_model_and_transforms, CLAP
        from inference import load_audio

        self.load_audio = load_audio

        config_path = af_dir / 'configs' / 'inference_af2_05b.yaml'
        with config_path.open() as f:
            config = yaml.safe_load(f)

        model_config = config['model_config']
        self.clap_config = config['clap_config']

        print(f'[AF2] Downloading/checking {self.model_id}', flush=True)
        ckpt_dir = Path(snapshot_download(repo_id=self.model_id))

        clap_path = ckpt_dir / 'clap_ckpt' / 'epoch_16.pt'
        if not clap_path.exists():
            raise FileNotFoundError(clap_path)

        self.clap_config['checkpoint'] = str(clap_path)

        original_module_to = torch.nn.Module.to

        def cpu_clap_to(module, *args, **kwargs):
            if module.__class__.__name__ == 'HTSAT_Swin_Transformer':
                return original_module_to(module, 'cpu')
            return original_module_to(module, *args, **kwargs)

        original_clap_init = CLAP.__init__

        def patched_clap_init(instance, clap_config):
            torch.nn.Module.to = cpu_clap_to
            try:
                original_clap_init(instance, clap_config)
            finally:
                torch.nn.Module.to = original_module_to

        CLAP.__init__ = patched_clap_init

        try:
            print('[AF2] Creating model on CPU', flush=True)
            self.model, self.tokenizer = create_model_and_transforms(
                **model_config,
                clap_config=self.clap_config,
                use_local_files=False,
                gradient_checkpointing=False,
                freeze_lm_embeddings=True,
            )

        finally:
            CLAP.__init__ = original_clap_init
            torch.nn.Module.to = original_module_to

        metadata_path = ckpt_dir / 'safe_ckpt' / 'metadata.json'
        with metadata_path.open() as f:
            metadata = json.load(f)

        for chunk_name in metadata:
            chunk_path = (ckpt_dir / 'safe_ckpt' / f'{chunk_name}.safetensors')

            print(f'[AF2] Loading {chunk_name}', flush=True)
            weights = load_file(str(chunk_path), device='cpu')

            incompatible = self.model.load_state_dict(weights, strict=False)

            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f'Unexpected checkpoint keys: '
                    f'{incompatible.unexpected_keys[:10]}'
                )

            del weights

        print('[AF2] Moving model to GPU (FP16)', flush=True)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()
        self.model.eval()

        self.yes_ids, self.no_ids = yes_no_ids(self.tokenizer)
        self.yes_t = torch.tensor(self.yes_ids, dtype=torch.long, device=self.device)
        self.no_t = torch.tensor(self.no_ids, dtype=torch.long, device=self.device)

        print('[AF2] Ready', flush=True)

    def build_conv(self, question, audio_path):
        return {
            'question': question,
            'audio_path': str(audio_path),
        }

    @torch.inference_mode()
    def _forward(self, convs):
        ps, lys, lns = [], [], []

        for conv in convs:
            audio, mask = self.load_audio(conv['audio_path'], self.clap_config)

            audio = audio.unsqueeze(0).to(self.device, dtype=self.dtype)
            mask = mask.unsqueeze(0).to(self.device, dtype=self.dtype)

            question = conv['question'].strip().lower()
            prompt = (f'<audio>{question}{self.tokenizer.sep_token}')

            encoded = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=512,
                truncation=True,
            )

            input_ids = encoded['input_ids'].to(self.device)

            with torch.autocast(
                device_type='cuda',
                dtype=self.dtype,
                enabled=(self.device.type == 'cuda'),
            ):
                outputs = self.model(
                    audio_x=audio,
                    audio_x_mask=mask,
                    lang_x=input_ids,
                    attention_mask=encoded['attention_mask'].to(
                        self.device
                    ),
                    use_cache=False,
                )

            logits = outputs.logits[:, -1, :]

            p, ly, ln = pyes_from_logits(logits, self.yes_t, self.no_t)
            ps.extend(p)
            lys.extend(ly)
            lns.extend(ln)

            del outputs, logits, audio, mask

        return ps, lys, lns
