import os
import random
import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Tanh
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock
from .distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from .W2V import Model
# from distortions.dl import distortion
import json
import torchaudio

DEFAULT_TIMBRE_SSL_SOURCE = os.environ.get("TIMBRE_SSL_SOURCE", "facebook/mms-300m")


def save_spectrum(spect, phase, flag='linear'):
    import numpy as np
    import os
    import librosa
    import librosa.display
    root = "draw_figure"
    import matplotlib.pyplot as plt
    spect = spect/torch.max(torch.abs(spect))
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    phase = phase/torch.max(torch.abs(phase))
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.clim(-40, 40)
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)

def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    import librosa.display
    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root,"feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap='gray')
        ax.axis('off')
        output_file = os.path.join(output_folder, f'feature_map_channel_{channel_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def save_waveform(a_tensor, flag='original'):
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile
    root = "draw_figure"
    y = a_tensor.cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=22050)
    D = librosa.stft(y)
    spectrogram = np.abs(D)
    img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)




def freeze_model_and_submodules(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.children():
        if isinstance(module, nn.Module):
            freeze_model_and_submodules(module)


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]

        #MLP for the input wm
        #stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

        self.ENc = Conv2Encoder(input_channel=1, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)

        self.EM = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)

        self.msg_processor = torch.nn.Embedding(msg_length, win_dim)

    def forward(self, x, msg, global_step, logit):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
#         watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        msg_aux = self.msg_processor(logit)
#         print(msg_aux)
        watermark_encoded = msg_aux.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked

    def test_forward(self, x, logit):
        num_samples = x.shape[2]
        spect, phase, _ = self.stft.transform(x)
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        msg_aux = self.msg_processor(logit.to(x.device))
        watermark_encoded = msg_aux.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked
    
    def save_forward(self, x, msg):
        num_samples = x.shape[2]
        save_waveform(x.squeeze())
        spect, phase, _ = self.stft.transform(x)
        # save spectrum
        save_spectrum(spect, phase, 'linear')
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        # save feature_map
        # save_feature_map(carrier_encoded[0])
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        save_spectrum(carrier_wateramrked.squeeze(1), phase, 'wmed_linear')
        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        save_waveform(y.squeeze().squeeze(), 'wmed')
        return y, carrier_wateramrked


class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.ssl = Model(1024, DEFAULT_TIMBRE_SSL_SOURCE, 12)

    def forward(self, y, global_step):
        msg, _ = self.ssl(y_d_d)
        msg_identity, _ = self.ssl(y)
        return msg.unsqueeze(0), msg_identity.unsqueeze(0)

    def test_forward(self, y, logit): 
        msg, _ = self.ssl(y)                                                        
        return msg


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                ReluBlock(1,16,3,1,1),
                ReluBlock(16,32,3,1,1),
                ReluBlock(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x):
        spect, phase, _ = self.stft.transform(x)
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
