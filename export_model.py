import torch
from torch import nn
from torchvision.models import resnet34

from torch.utils.mobile_optimizer import optimize_for_mobile, MobileOptimizerType
import torch.nn.utils.prune as prune
import metafg_model
from predict_single_file import image_proprecess

# model = metafg_model.get_metafg_model()
# torch.save(model.state_dict(), 'model20240824.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet34()
model.fc = nn.Linear(model.fc.in_features, 11000)
model.load_state_dict(torch.load('model20240824.pth'))
model = model.to(device)
model.eval()

img, data = image_proprecess("test_images/86810585ED8E85C2CE8525BB8E17CF07.jpg")
example = data.to(device)

traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module,
                                                 {
                                                     MobileOptimizerType.CONV_BN_FUSION, # I'm only disabling CONV_BN_FUSION
                                                      # MobileOptimizerType.FUSE_ADD_RELU,
                                                      # MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
                                                      # MobileOptimizerType.INSERT_FOLD_PREPACK_OPS,
                                                      # MobileOptimizerType.REMOVE_DROPOUT
                                                 })
optimized_traced_model._save_for_lite_interpreter("model20240824-2.pt")