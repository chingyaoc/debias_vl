import torch
from torch import autocast
import torch.nn.functional as F
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from diffusers import LMSDiscreteScheduler

from sklearn.decomposition import TruncatedSVD
import argparse
import os

from job_list import train_list, test_list


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debiased Diffusion Models')
    parser.add_argument('--cls', default="doctor", type=str, help='target class name')
    parser.add_argument('--lam', default=500, type=float, help='regualrizer constant')
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    args = parser.parse_args()


    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )

    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )


    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    scheduler = LMSDiscreteScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )


    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device) 

    # Construct Positive Pair
    candidate_prompt = []
    S = []
    counter = 0
    for train_cls_i in train_list:
        train_cls_i = train_cls_i.lower()
        candidate_prompt += ['A photo of a male {}.'.format(train_cls_i), 'A photo of a female {}.'.format(train_cls_i)]
        S += [[counter, counter + 1]]
        counter += 2

    candidate_input = tokenizer(candidate_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        candidate_embeddings = text_encoder(candidate_input.input_ids.to(torch_device))[0]#.cpu().numpy()
    candidate_embeddings = candidate_embeddings[torch.arange(candidate_embeddings.shape[0]), candidate_input['input_ids'].argmax(-1)]
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1).cpu().numpy()

    # Compute Calibration Matrix
    M =  get_M(candidate_embeddings, S)
    G = args.lam * M + np.eye(M.shape[0]) 
    P = np.linalg.inv(G)
    P = torch.tensor(P).cuda()


    # Language Prompt
    prompt = ["A photo of a " + args.cls + "."]
    print("Prompt: {}".format(prompt))

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 100           # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(12345)   # Seed generator to create the inital latent noise
    batch_size = 1


    # Define Text Embedding
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   

    # Debias Text Embedding
    text_embeddings = torch.matmul(text_embeddings, P.T.float())
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    # Diffusion Sampling
    save_dir = "output_" + args.cls + "_lam" + str(args.lam)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    for i in tqdm(range(100)):
        # Generate Initial Noise
        latents = torch.randn(
                   (batch_size, unet.in_channels, height // 8, width // 8),
                   generator=generator,
                  )
        latents = latents.to(torch_device)


        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma


        counter = 0
        for t in tqdm(scheduler.timesteps):
          # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
          latent_model_input = torch.cat([latents] * 2)

          latent_model_input = scheduler.scale_model_input(latent_model_input, t)

          # predict the noise residual
          with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

          # perform guidance
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

          # compute the previous noisy sample x_t -> x_t-1
          latents = scheduler.step(noise_pred, t, latents).prev_sample


        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = vae.decode(latents).sample


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save(save_dir+"/img_{}_{}.jpg".format(args.cls, i))

