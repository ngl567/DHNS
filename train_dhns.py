import mmkgc
import torch
from mmkgc.config import Trainer_dhns, Tester_dhns
from mmkgc.module.model import AdvMixDistMult
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling_complex
from mmkgc.data import TrainDataLoader_complex, TestDataLoader_complex
from mmkgc.adv.mmmodules_distmult import DiffHEG

# dataloader for training
train_dataloader = TrainDataLoader_complex(
	in_path = "./benchmarks/MKG-Y/", 
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader_complex("./benchmarks/MKG-Y/", "link")

img_emb = torch.load('./embeddings/MKG-Y-visual.pth')
text_emb = torch.load('./embeddings/MKG-Y-textual.pth')

# define the model
distmult = AdvMixDistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024,
	margin = 200.0,
	epsilon = 2.0,
	img_emb=img_emb,
	text_emb=text_emb
)

# define the loss function
model = NegativeSampling_complex(
	model = distmult, 
	loss = SigmoidLoss(adv_temperature = 0.5),
	batch_size = train_dataloader.get_batch_size(), 
	l3_regul_rate = 0.000005
)

adv_generator = DiffHEG(
        embedding_dim=1024,
        T=50,
        dim_r=1024,
        margin=200.0,
        eps=2.0
    )

# train the model
trainer = Trainer_dhns(model = model, data_loader = train_dataloader, train_times = 400, alpha = 0.002, use_gpu = True, opt_method = "adam",
        generator=adv_generator,
        lrg=0.002,
        mu=0.01,
        g_epoch=10)
trainer.run()
distmult.save_checkpoint('./checkpoint/distmult.ckpt')

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester_dhns(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
