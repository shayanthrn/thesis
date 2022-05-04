import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from parallel_wavegan.utils import load_model
from .StarGAN.Utils.ASR.models import ASRCNN
from .StarGAN.Utils.JDC.model import JDCNet
from .StarGAN.models import Generator, MappingNetwork, StyleEncoder
import soundfile as sf



class Inference():

    def __init__(self):
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        # load F0 model
        self.F0_model = JDCNet(num_class=1, seq_len=192)
        params = torch.load("./VoiceConversion/StarGAN/Utils/JDC/bst.t7")['net']
        self.F0_model.load_state_dict(params)
        _ = self.F0_model.eval()
        self.F0_model = self.F0_model.to(self.device)
        # load vocoder
        self.vocoder = load_model("./Vocoder/checkpoint-400000steps.pkl").eval().to(self.device)
        self.vocoder.remove_weight_norm()
        _ = self.vocoder.eval()
        # load starganv2
        model_path = './models/epoch_00148.pth'
        with open('./models/config.yml') as f:
            starganv2_config = yaml.safe_load(f)
        self.starganv2 = self.build_model(model_params=starganv2_config["model_params"])
        params = torch.load(model_path, map_location='cpu')
        params = params['model_ema']
        _ = [self.starganv2[key].load_state_dict(params[key]) for key in self.starganv2]
        _ = [self.starganv2[key].eval() for key in self.starganv2]
        # self.starganv2.style_encoder = self.starganv2.style_encoder.to(self.device)
        self.starganv2.mapping_network = self.starganv2.mapping_network.to(self.device)
        self.starganv2.generator = self.starganv2.generator.to(self.device)

    def inference(self, sourcefile, speaker, targetfile):
        # load input wave
        audio, source_sr = librosa.load(sourcefile, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32
        # no reference, using mapping network
        speaker_dicts = {}
        index= int(speaker)-1
        speaker = 'p' + speaker

        speaker_dicts[speaker] = ('', index)
        reference_embeddings = self.compute_style(speaker_dicts,self.starganv2)

        source = self.preprocess(audio).to(self.device)
        keys = []
        converted_samples = {}
        reconstructed_samples = {}
        converted_mels = {}

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = self.F0_model.get_feature_GAN(source.unsqueeze(1))
                out = self.starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
                c = out.transpose(-1, -2).squeeze().to(self.device)
                y_out = self.vocoder.inference(c)
                y_out = y_out.view(-1).cpu()
                if key not in speaker_dicts or speaker_dicts[key][0] == "":
                    recon = None
                else:
                    wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                    mel = self.preprocess(wave)
                    c = mel.transpose(-1, -2).squeeze().to(self.device)
                    recon = self.vocoder.inference(c)
                    recon = recon.view(-1).cpu().numpy()
            converted_samples[key] = y_out.numpy()
            reconstructed_samples[key] = recon
            converted_mels[key] = out
            keys.append(key)
        
        for key, wave in converted_samples.items():
            sf.write(targetfile, wave, 24000)
        return True


    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def build_model(self, model_params={}):
        args = Munch(model_params)
        generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
        mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
        style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
        
        nets_ema = Munch(generator=generator,
                        mapping_network=mapping_network,
                        style_encoder=style_encoder)

        return nets_ema

    def compute_style(self, speaker_dicts,starganv2):
        reference_embeddings = {}
        for key, (path, speaker) in speaker_dicts.items():
            if path == "":
                label = torch.LongTensor([speaker]).to(self.device)
                latent_dim = starganv2.mapping_network.shared[0].in_features
                ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(self.device), label)
            else:
                wave, sr = librosa.load(path, sr=24000)
                audio, index = librosa.effects.trim(wave, top_db=30)
                if sr != 24000:
                    wave = librosa.resample(wave, sr, 24000)
                mel_tensor = self.preprocess(wave).to(self.device)

                with torch.no_grad():
                    label = torch.LongTensor([speaker])
                    ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
            reference_embeddings[key] = (ref, label)
        
        return reference_embeddings