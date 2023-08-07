# DiffPure extension for Auto1111 by kabachuha

import os, yaml
import sys, time
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from types import SimpleNamespace
import gradio as gr
from modules import script_callbacks, shared
from modules import sd_hijack, lowvram
from modules.shared import cmd_opts, opts, devices
import modules.scripts as scripts

from modules import processing
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, create_infotext
from modules.shared import sd_model, opts
import modules.images as images
import modules.paths as ph

from torchvision import transforms
from torchvision.transforms.transforms import Resize

# Fix system paths

folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui')  # for TheLastBen's colab
for _ in basedirs:
    paths_to_ensure = [
        os.path.join(folder_name, 'scripts'),
        os.path.join(folder_name, 'scripts', 'diffpure_core')
    ]
    for scripts_path_fix in paths_to_ensure:
        if scripts_path_fix not in sys.path:
            sys.path.extend([scripts_path_fix])

def get_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if ext.name in ["sd-webui-diffpure"] and ext.enabled:
                return ext.version
        return "Unknown"
    except:
        return "Unknown"



# DiffPure part

# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
#from runners.diffpure_sde import RevGuidedDiffusion
#from runners.diffpure_ode import OdeGuidedDiffusion
#from runners.diffpure_ldsde import LDGuidedDiffusion

class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device, model_dir=os.path.join(ph.models_path, 'DiffPure'))
        #elif args.diffusion_type == 'sde':
        #    self.runner = RevGuidedDiffusion(args, config, device=config.device)
        #elif args.diffusion_type == 'ode':
        #    self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        #elif args.diffusion_type == 'ldsde':
        #    self.runner = LDGuidedDiffusion(args, config, device=config.device)
        #elif args.diffusion_type == 'celebahq-ddpm':
        #    self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.device = config.device
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = (x_re + 1) * 0.5

        self.counter += 1

        return out

# Now back to the extension

## Gradio script declaration

class Script(scripts.Script):

    def title(self):
        return "DiffPure (adversarial noise purification)"
    
    def show(self, is_img2img):
        return is_img2img
    
    def ui(self, is_img2img):
        with gr.Accordion(label='Info & License', open=False):
            gr.Markdown('Put the pretrained models (for example, [this one for ImageNet](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)) into models/DiffPure, also put there [the corresponding .yaml file](https://github.com/NVlabs/DiffPure/blob/master/configs/imagenet.yml)')
            gr.Markdown('Please check the [LICENSE](https://github.com/NVlabs/DiffPure/blob/master/LICENSE) file. This work may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please contact researchinquiries@nvidia.com.')

    def run(self, p):
        print(f"😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳")
        print(f"😳\033[4;33m DiffPure extension for Auto1111's webui\033[0m😳")
        print(f"😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳😳")
        print("Initialazing")

        if shared.sd_model is not None:
            print('Knocking out Stable Diffusion 😎')
            print("Non-latent diffusion models are heccin' chonkers, pray you have enough vram")
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)
            try:
                lowvram.send_everything_to_cpu()
            except Exception as e:
                pass
            # the following command actually frees the GPU vram from the sd.model, no need to do del shared.sd_model 22-05-23
            shared.sd_model = None
            print('SD unloaded')
        gc.collect()
        devices.torch_gc()

        print('Starting the model and loader...')

        ngpus = torch.cuda.device_count()
        adv_batch_size = 1 # p.batch_size
        # print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

        pics = p.init_images

        print('Reading the config you put in the models folder')

        models_path = os.path.join(ph.models_path, 'DiffPure')
        cfg_file = None

        # Just grab the first one
        for f in os.listdir(models_path):
            if f.endswith('.yaml') or f.endswith('.yml'):
                cfg_file = os.path.join(models_path, f)
                break
        assert cfg_file is not None

        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config = SimpleNamespace(**config)
        config.device = devices.device

        print('Loading the model')

        args = {}
        args['config'] = cfg_file
        args['data_seed'] = 0
        args['seed'] = 1234
        args['exp'] = 'exp'
        args['verbose'] = 'info'
        args['image_folder'] = 'images'
        args['ni'] = False
        args['sample_step'] = 1
        args['t'] = 400
        args['t_delta'] = 15
        args['rand_t'] = False
        args['diffusion_type'] = 'ddpm'
        args['score_type'] = 'guided_diffusion'
        args['eot_iter'] = 20
        args['use_bm'] = False

        args['sigma2'] = 1e-3
        args['lambda_ld'] = 1e-2
        args['eta'] = 5.
        args['step_size'] = 1e-3

        args['domain'] = 'imagenet'
        args['classifier_name'] = 'Eyeglasses'
        args['partition'] = 'val'
        args['adv_batch_size'] = 64
        args['attack_type'] = 'square'
        args['lp_norm'] = 'Linf'
        args['attack_version'] = 'custom'

        args['num_sub'] = 1000
        args['adv_eps'] = 0.07

        args = SimpleNamespace(**args)

        model = SDE_Adv_Model(args, config)

        print('Sending to CUDA')

        model = model.eval().to(config.device)

        print('Model loaded! (at least I hope so)')

        print('Transforming the images to a suitable format')

        image_size = config.model['image_size']

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        output_images = []

        for pic in tqdm(pics, 'Processing pics'):
            # PIL -> Tensor
            img = transform((pic))

            img = model(img)

            transform_back = transforms.ToPILImage()

            img = transform_back(img)

            output_images += img

        print('Processed! Have a cake 🥰')

        processed = Processed(p, output_images, p.seed, "p_info")
        return processed
