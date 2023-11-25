import torch
import torchaudio
import io, os
import numpy as np
from audio_feature_extraction import LFCC, repeat_padding_RawTensor
import torch.nn.functional as F


class ASVspoofHandler(object):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        # Sampling rate for ASVspoof model must be 16k
        self.expected_sampling_rate = 16000
        self.feat_len = 120000

    def initialize(self, context):
        """Initialize properties and load model"""
        self._context = context
        properties = context.system_properties

        # See https://pytorch.org/serve/custom_service.html#handling-model-execution-on-multiple-gpus
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        model_dir = properties.get("model_dir")
        serialized_file = context.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True

    def _load_model(self, model_path):
        self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.eval()

    def preprocess(self, featureTensor):
        lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
        this_feat_len = featureTensor.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len - self.feat_len)
            featureTensor = featureTensor[:, startp:startp + self.feat_len]
        elif this_feat_len < self.feat_len:
            featureTensor = repeat_padding_RawTensor(featureTensor, self.feat_len)
        featureTensor = lfcc(featureTensor)
        return featureTensor

    def handle(self, data, context):
        """Transform input to tensor, resample, run model and return transcribed text."""
        input = data[0].get("data")
        if input is None:
            input = data[0].get("body")

        # torchaudio.load accepts file like object, here `input` is bytes
        model_input, sample_rate = torchaudio.load(io.BytesIO(input))

        # Ensure sampling rate is the same as the trained model
        if sample_rate != self.expected_sampling_rate:
            model_input = torchaudio.functional.resample(model_input, sample_rate, self.expected_sampling_rate)

        ## get input tensor is done, now doing preprocessing
        model_input = self.preprocess(model_input)

        ## inference
        feat = model_input.to(self.device).unsqueeze(0).transpose(2, 3)
        feats, feat_outputs = self.model(feat)

        # score0 = F.softmax(feat_outputs)[:, 0].item()
        # score1 = F.softmax(feat_outputs)[:, 1].item()

        score0 = feat_outputs[:, 0].item()

        # ## get output
        # if score0 > score1:
        #     return ["bonafide\n"]
        # else:
        #     return ["spoofing\n"]
        return ["{\"score\": %.3f}\n" %score0]
