from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import argparse
import os
import logging


def get_A(z_i, z_j, w):
    z_i = z_i[:, None] * w[0]
    z_j = z_j[:, None] * w[1]
    return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))

def get_M(embeddings, S, w):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]], [w[s[0]], w[s[1]]])
    return M / len(S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debiased Diffusion Models')
    parser.add_argument('--cls', default="doctor", type=str, help='target class name')
    parser.add_argument('--lam', default=100, type=float, help='regualrizer constant')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()


    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    model = model.to(torch_device)
    text_encoder = text_encoder.to(torch_device)

    # 3. The UNet model for generating the latents.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)


    candidate_prompt = ['A photo of a male.', 'A photo of a female.']
    S = [[0, 1]]
    candidate_input = tokenizer(candidate_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        candidate_embeddings = text_encoder(candidate_input.input_ids.to(torch_device))[0]#.cpu().numpy()
    candidate_embeddings = candidate_embeddings[torch.arange(candidate_embeddings.shape[0]), candidate_input['input_ids'].argmax(-1)]
    candidate_embeddings = candidate_embeddings.cpu().numpy()


    f_name = "gender_" + args.cls + "_lam" + str(args.lam) + "_lr" + str(args.lr)
    logging.basicConfig(filename="logs/" + f_name + ".log", encoding='utf-8', level=logging.DEBUG)
    save_dir = "diffusion_outputs/" + f_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    w = np.ones(2)
    BS = 100
    GS = 2
    lr = args.lr
    for t in range(50):
        print("===== Current w =====")
        print(w)
        print("=====================")

        logging.debug("Current w at step {}".format(t))
        logging.debug(w)

        # Solve projection matrix
        P0 = np.eye(candidate_embeddings.shape[1])
        M =  get_M(candidate_embeddings, S, w)
        G = args.lam * M + np.eye(M.shape[0])
        P = np.matmul(P0, np.linalg.inv(G))
        P = torch.tensor(P).cuda()

        # generate image (Batch size 100)
        if args.cls[0] in ['a', 'e', 'i', 'o', 'u']:
            prompt = ["A photo of an " + args.cls + "."]
        else:
            prompt = ["A photo of a " + args.cls + "."]
        print("Prompt: {}".format(prompt))
        logging.debug("Prompt: {}".format(prompt))

        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = 100           # Number of denoising steps
        guidance_scale = 7.5                # Scale for classifier-free guidance
        generator = torch.manual_seed(np.random.randint(1e4))
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

        ##### DEBIAS #####
        text_embeddings = torch.matmul(text_embeddings, P.T.float())
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        for i in tqdm(range(BS)):
            # Generate Initial Noise
            latents = torch.randn(
                       (batch_size, unet.in_channels, height // 8, width // 8),
                       generator=generator,
                      )
            latents = latents.to(torch_device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
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
            pil_images[0].save(save_dir+"/img_{}.jpg".format(i))




        # Evaluate the generation
        preds = []
        for i in tqdm(range(BS)):
            image = Image.open(save_dir+"/img_{}.jpg".format(i))
            inputs = processor(text=["A photo of a male.", "A photo of a female."], images=image, return_tensors="pt", padding=True)
            inputs = inputs.to(torch_device)

            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)
            preds.append(probs)

        preds = torch.stack(preds).squeeze(1)
        prob1 = preds[:, 0].cpu().numpy()
        pred = (prob1 > 0.5).astype(int)
        num_pred_0 = sum(pred)
        num_pred_1 = BS - num_pred_0

        r0 = num_pred_0 / BS
        r1 = num_pred_1 / BS
        print("r0: {}, r1: {}".format(r0, r1))
        g0 = r0 - 1 / GS
        g1 = r1 - 1 / GS
        w[0] = w[0] + lr * g0
        w[1] = w[1] + lr * g1

        logging.debug("r0: {}, r1: {}".format(r0, r1))

