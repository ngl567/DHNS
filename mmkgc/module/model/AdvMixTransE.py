import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class AdvMixTransE(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        p_norm=1,
        norm_flag=True
    ):

        super(AdvMixTransE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        if margin == None or epsilon is None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_proj = nn.Linear(self.img_dim, self.dim)
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(True)
        self.text_proj = nn.Linear(self.text_dim, self.dim)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)
        self.ent_attn = nn.Linear(self.dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        self.p_norm = p_norm
        self.norm_flag = norm_flag
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
        

    def get_joint_embeddings(self, es, ev, et):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors
    

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score
    

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb)
        
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score
    
    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)
    
    def get_batch_rel_embs(self, data):
        return self.rel_embeddings(data)
    
    def get_batch_img_embs(self, data):
        return self.img_proj(self.img_embeddings(data))
    
    def get_batch_text_embs(self, data):
        return self.text_proj(self.text_embeddings(data))
    
    
    def mm_negative_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        w_margin,
        neg_h=None,
        neg_t=None,
        neg_hv=None, 
        neg_tv=None,
        neg_ht=None,
        neg_tt=None
    ):
        if neg_hv is None or neg_tv is None or neg_ht is None or neg_tt is None:
            raise NotImplementedError
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = batch_r
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb).unsqueeze(1)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb).unsqueeze(1)
        h_neg = self.get_joint_embeddings(neg_h, neg_hv, neg_ht)
        t_neg = self.get_joint_embeddings(neg_t, neg_tv, neg_tt)

        score_h = w_margin * self.margin - self._calc(h_neg, t_joint, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        score_t = w_margin * self.margin - self._calc(h_joint, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        score_all = w_margin * self.margin - self._calc(h_neg, t_neg, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        
        return [score_h, score_t, score_all], [h_img_emb, t_img_emb, h_text_emb, t_text_emb]

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul
    
    def get_attention(self, es, ev, et):
        
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights

    def get_attention_weight(self, h, t):
        h = torch.LongTensor([h])
        t = torch.LongTensor([t])
        h_s = self.ent_embeddings(h)
        t_s = self.ent_embeddings(t)
        h_img_emb = self.img_proj(self.img_embeddings(h))
        t_img_emb = self.img_proj(self.img_embeddings(t))
        h_text_emb = self.text_proj(self.text_embeddings(h))
        t_text_emb = self.text_proj(self.text_embeddings(t))
        
        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb)
        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb)
        return h_attn, t_attn
