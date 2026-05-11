
import torch
from tqdm import tqdm

from nrm.model import MLP
from nrm.dataset.loader import ValidationSet

from datetime import datetime
from datetime import timedelta


device = torch.device("cuda")
model_id = 13

validation_set = ValidationSet(1000, False, "test_numerical")

model = MLP.from_id(model_id).to(device)
loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

model.eval()
times = []
for batch_idx, (morph, pose, _) in enumerate(tqdm(validation_set, desc=f"Validation")):
    morph = morph.to(device, non_blocking=True)
    pose = pose.to(device, non_blocking=True)
    start = datetime.now()
    logit = model.predict(morph, pose)
    times += [datetime.now() - start]
avg = sum(times, timedelta(0)) / len(times)
print((avg.seconds * 10**6 + avg.microseconds) / len(validation_set))

