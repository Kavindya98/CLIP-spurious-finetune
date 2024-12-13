import numpy as np
from collections import OrderedDict
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import models.slip.slip_models as models
import models.slip.slip_utils as utils
from models.slip.slip_tokenizer import SimpleTokenizer
import open_clip

from transformers import (
    CLIPImageProcessor, BlipImageProcessor, XCLIPProcessor, EfficientNetImageProcessor,
    CLIPTokenizerFast, BertTokenizerFast, CLIPTokenizer,
    CLIPModel, BlipModel, FlavaModel, XCLIPModel, Blip2Model, AlignModel,
    Pix2StructTextModel, FlavaTextModel,
    Pix2StructVisionModel, FlavaImageModel,
    FlavaMultimodalModel, FlavaImageProcessor,
    BridgeTowerImageProcessor, RobertaTokenizerFast, BridgeTowerModel,
    Blip2Processor, Blip2ForConditionalGeneration, BlipForConditionalGeneration,
    Pix2StructForConditionalGeneration
)
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from models.alip import (
    create_model, tokenize, factory
)

from models.clip import imagenet_templates

class ZeroShotVLM(nn.Module):
    def __init__(self, model, metadata_map, templates=imagenet_templates, tokenizer=None):
        super(ZeroShotVLM, self).__init__()
        self.model = model
        self.tokenizer = tokenizer       
        self.metadata_map = metadata_map
        self.classnames = metadata_map['y']
        self.templates = templates
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained VLM models.
        """
        raise NotImplementedError
    
    def featurizer(self, input):
        raise NotImplementedError
    
    def zeroshot_weights(self):
        return self.zeroshot_classifier(self.classnames)

    def classifier(self, input_features):
        zeroshot_weights = self.zeroshot_weights()
        logits = self.logit_scale.exp() * input_features @ zeroshot_weights
        return logits

    def forward(self, input, return_features=False):
        input_features = self.featurizer(input)
        logits = self.classifier(input_features)
        if return_features:
            return logits, input_features
        else:
            return logits
    
    def train(self, mode=True):
        self.model.train(mode)


class SLIP_ZeroShot(ZeroShotVLM):
    def __init__(self, metadata_map, templates=imagenet_templates):
        ckpt = torch.load('models/weights/slip_base_100ep.pt')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        
        # create model
        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))
        model = getattr(models, old_args.model)(rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.load_state_dict(state_dict, strict=True)

        tokenizer = SimpleTokenizer()

        super().__init__(model, metadata_map, templates, tokenizer)

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained SLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = self.tokenizer(texts).cuda() #tokenize
            class_embeddings =  utils.get_model(self.model).encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def featurizer(self, input):
        input_features = utils.get_model(self.model).encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features


def alip_get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed

class ALIP_ZeroShot(ZeroShotVLM):
    def __init__(self, metadata_map, templates=imagenet_templates):
        model = create_model('ViT-B/32')
        state_dict = alip_get_state_dict('models/weights/ALIP_YFCC15M_B32.pt')
        model.load_state_dict(state_dict, strict=True)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        super().__init__(model, metadata_map, templates, tokenizer)

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained SLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = self.tokenizer(texts).cuda() #tokenize
            class_embeddings =  self.model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features
    
class LaClip_ZeroShot(ZeroShotVLM):
    def __init__(self, metadata_map, templates=imagenet_templates):
        ckpt = torch.load('models/weights/laion400m_laclip_vitb16.pt')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16',
        '',
        precision='amp',
        device='cuda',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=224,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
        )
        model.load_state_dict(state_dict, strict=True)
        tokenizer = SimpleTokenizer()
        super().__init__(model, metadata_map, templates, tokenizer)

    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained SLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = self.tokenizer(texts).cuda() #tokenize
            class_embeddings =  self.model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

class Flava_ZeroShot(nn.Module):
    def __init__(self, metadata_map, templates=imagenet_templates):
        super(Flava_ZeroShot, self).__init__()
        self.model_image = FlavaImageModel.from_pretrained("facebook/flava-full")
        self.model_text = FlavaTextModel.from_pretrained("facebook/flava-full")
        self.metadata_map = metadata_map
        self.classnames = metadata_map['y']
        self.templates = templates
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.tokenizer =  BertTokenizerFast.from_pretrained('bert-base-uncased')
        

    def featurizer(self, input):
        
        input_features = self.model_image(input.squeeze(1)).pooler_output
        
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained SLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt") #tokenize
            texts = {k: torch.LongTensor(np.array(v)).cuda() for k, v in texts.items()}
            class_embeddings =  self.model_text(**texts).pooler_output #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def zeroshot_weights(self):
        return self.zeroshot_classifier(self.classnames)

    def classifier(self, input_features):
        zeroshot_weights = self.zeroshot_weights()
        logits = self.logit_scale.exp() * input_features @ zeroshot_weights
        return logits

    def forward(self, input, return_features=False):
        input_features = self.featurizer(input)
        logits = self.classifier(input_features)
        if return_features:
            return logits, input_features
        else:
            return logits
    
    def train(self, mode=True):
        self.model_image.train(mode)
        self.model_text.train(mode)



class BLIP_ZeroShot(ZeroShotVLM):
    def __init__(self, metadata_map, templates=imagenet_templates):
        model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        super().__init__(model, metadata_map, templates, tokenizer)

    def featurizer(self, input):
        input_features = self.model.get_image_features(input.squeeze(1)).pooler_output
        print("Shape of input_features:", input_features.shape)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        
        return input_features

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained SLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt") #tokenize
            texts = {k: torch.LongTensor(np.array(v)).to('cuda') for k, v in texts.items()}
            class_embeddings =  self.model.get_text_features(**texts).logits #embed with text encoder
            print("Shape of class_embeddings before normalization:", class_embeddings.shape)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    