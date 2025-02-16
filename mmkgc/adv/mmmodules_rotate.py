import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Denoiser
class CEDenoiser(nn.Module):
    def __init__(self, embedding_dim, dim_r, margin, eps):
        super(CEDenoiser, self).__init__()
        self.scoring_module = ScoringModule(dim_r, margin, eps)
        self.denoiser_block = CEDenoiserBlock(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x_t, t, x_ht, x_r, mode):
        c = self.scoring_module(x_ht, x_r, mode)
        e = self.denoiser_block(x_t, t, c)
        noise_pred = self.output_layer(e)
        return noise_pred

# Scoring Module for obtain condition
class ScoringModule(nn.Module):
    def __init__(self, dim_r, margin, eps):
        super(ScoringModule, self).__init__()
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(margin + eps) / dim_r]),
            requires_grad=False
        )        

    def forward(self, ht, r, mode):
        pi = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        pi.requires_grad = False

        re_ht, im_ht = torch.chunk(ht, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi).to('cuda')

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "tail":
            re_cond = re_ht * re_relation - im_ht * im_relation
            im_cond = re_ht * im_relation + im_ht * re_relation
        else:
            re_cond = re_relation * re_ht + im_relation * im_ht
            im_cond = re_relation * im_ht - im_relation * re_ht

        cond = torch.cat([re_cond, im_cond], dim=-1)
        cond = cond.squeeze(1)

        return cond


class SinusoidalPosEmb(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        half_dim = (self.embedding_dim // 2)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device='cuda') * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1).squeeze(1)
        return emb[:,:self.embedding_dim]
    

# Denoiser block
class CEDenoiserBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(CEDenoiserBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim*3)
        self.mlp1 = nn.Linear(embedding_dim*3, embedding_dim * 4)
        self.layer_norm2 = nn.LayerNorm(embedding_dim * 4)
        self.mlp2 = nn.Linear(embedding_dim * 4, embedding_dim)

        sinu_pos_emb = SinusoidalPosEmb(embedding_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x_t, t, c):
        t = self.time_mlp(t)
        x = torch.cat([x_t, t, c], dim=-1)
        x = self.layer_norm1(x)
        x = F.relu(self.mlp1(x))
        x = self.layer_norm2(x)
        x = self.mlp2(x)
        return x


# Diffusion-based Multi-Level Embedding Generation
class DiffHEG(nn.Module):
    def __init__(self, embedding_dim, T, dim_r, margin, eps):
        super(DiffHEG, self).__init__()
        self.denoiser = CEDenoiser(embedding_dim, dim_r, margin, eps)
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T).to('cuda')
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to('cuda')
        self.embedding_dim = embedding_dim

    def forward(self, x_h, x_r, x_t):

        t_tensor = torch.randint(1, self.T + 1, (x_h.shape[0],)).to('cuda')
        t_tensor = t_tensor.unsqueeze(-1)

        epsilon = torch.randn_like(x_t)

        x_t_noisy = self.q_sample(x_t, t_tensor, epsilon)
        t_noise_pred = self.denoiser(x_t_noisy, t_tensor, x_h, x_r, 'tail')
        
        x_h_noisy = self.q_sample(x_h, t_tensor, epsilon)
        h_noise_pred = self.denoiser(x_h_noisy, t_tensor, x_t, x_r, 'head')

        loss = self.denoiser_loss(epsilon, t_noise_pred)
        loss += self.denoiser_loss(epsilon, h_noise_pred)

        return loss

    def q_sample(self, x_t, t, epsilon):
        alpha_bar_t = self.alpha_bars[t - 1]
        x_t_noisy = torch.sqrt(alpha_bar_t) * x_t + torch.sqrt(1 - alpha_bar_t) * epsilon
        return x_t_noisy

    def p_sample(self, x_noisy, t, x_ht, x_r, mode):
        beta_t = self.betas[t - 1]
        alpha_t = self.alphas[t - 1]
        alpha_bar_t = self.alpha_bars[t - 1]

        noise_pred = self.denoiser(x_noisy, t, x_ht, x_r, mode)

        mu = 1 / torch.sqrt(alpha_t) * (x_noisy - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred)
        sigma = torch.sqrt(beta_t)

        if t[0] > 1:
            z = torch.randn_like(x_noisy)
        else:
            z = 0

        x_t_prev = mu + sigma * z
        return x_t_prev

    def denoiser_loss(self, noise, noise_pred):
        loss = F.l1_loss(noise, noise_pred, reduction='none')
        return loss.mean()


    # denoise to generate the entity for multi-level negative triples generation
    @torch.no_grad()
    def sample(self, x_h, x_r, x_t):
        # start from pure noise (for each example in the batch)
        embs_t = []
        embs_h = []

        x_t_noisy = torch.randn((x_h.shape[0], self.embedding_dim)).to('cuda')

        for t in range(self.T, 0, -1):
            t_tensor = torch.full((x_h.shape[0], 1), t).to('cuda')
            x_t_noisy = self.p_sample(x_t_noisy, t_tensor, x_h, x_r, 'tail')
            x_h_noisy = self.p_sample(x_t_noisy, t_tensor, x_t, x_r, 'head')
            embs_t.append(x_t_noisy)
            embs_h.append(x_h_noisy)

        embs_h = embs_h[::-1]
        embs_t = embs_t[::-1]
        steps = [0, int(self.T/10), int(self.T/8), int(self.T/4), int(self.T/2)]
        
        out_t = [embs_t[step] for step in steps]
        out_h = [embs_h[step] for step in steps]
        return out_h, out_t 
    

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, node_emb, img_emb):
        batch_sim = self.sim_func(node_emb.unsqueeze(1), img_emb.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)