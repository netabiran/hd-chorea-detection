import torch
import torch.nn as nn

WIN_SIZE = 300

class SegModel(nn.Module):
    def __init__(self, pretrained_model, multi_windows=True, feat_dim=0, num_classes=5):
        super(SegModel, self).__init__()
        self.multi_windows = multi_windows
        self.pretrained_model = pretrained_model
        self.feat_dim = feat_dim
        
        factor = 3 if multi_windows else 1
        self.factor = factor
        
        self.upsample5 = nn.Upsample(scale_factor=5)
        self.upsample4 = nn.Upsample(scale_factor=5)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv5 = nn.Conv1d(factor * 1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv4 = nn.Conv1d((factor + 1) * 512, 256, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv1d((factor + 1) * 256, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        self.conv2 = nn.Conv1d((factor + 1) * 128, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        
        self.conv_out1 = nn.Conv1d((factor + 1) * 64, 32, kernel_size=5, stride=1, padding=2)
        self.conv_out2 = nn.Conv1d(64 if feat_dim else 32, num_classes, kernel_size=5, stride=1, padding=2)
        self.relu_out = nn.ReLU()

        # Only create feature layers if feat_dim > 0
        if feat_dim > 0:
            self.feat_fc1 = nn.Linear(feat_dim, 64)
            self.feat_fc2 = nn.Linear(64, 32)
            self.relu_feat = nn.ReLU()

    def forward(self, x, features=None):
        feature_extractor = self.pretrained_model.feature_extractor
        layer1_list, layer2_list, layer3_list, layer4_list, layer5_list = [], [], [], [], []
        
        for i in range(self.factor):
            layer1_list.append(self.get_layer_internal_result(feature_extractor.layer1, x[:, :, i * WIN_SIZE:(i + 1) * WIN_SIZE]))
            layer2_list.append(self.get_layer_internal_result(feature_extractor.layer2, layer1_list[i][-1]))
            layer3_list.append(self.get_layer_internal_result(feature_extractor.layer3, layer2_list[i][-1]))
            layer4_list.append(self.get_layer_internal_result(feature_extractor.layer4, layer3_list[i][-1]))
            layer5_list.append(self.get_layer_internal_result(feature_extractor.layer5, layer4_list[i][-1]))

        layer5_out = torch.cat([out[-2] for out in layer5_list], dim=1)
        layer4_out = torch.cat([out[-2] for out in layer4_list], dim=1)
        layer3_out = torch.cat([out[-2] for out in layer3_list], dim=1)
        layer2_out = torch.cat([out[-2] for out in layer2_list], dim=1)
        layer1_out = torch.cat([out[-2] for out in layer1_list], dim=1)

        skip_con5 = self.upsample5(layer5_out)
        conv5 = self.relu5(self.conv5(skip_con5))
        concat4 = torch.cat([layer4_out, conv5], dim=1)

        skip_con4 = self.upsample4(concat4)
        conv4 = self.relu4(self.conv4(skip_con4))
        concat3 = torch.cat([layer3_out, conv4], dim=1)

        skip_con3 = self.upsample3(concat3)
        conv3 = self.relu3(self.conv3(skip_con3))
        concat2 = torch.cat([layer2_out, conv3], dim=1)

        skip_con2 = self.upsample2(concat2)
        conv2 = self.relu2(self.conv2(skip_con2))
        concat1 = torch.cat([layer1_out, conv2], dim=1)

        out = self.relu_out(self.conv_out1(concat1))

        # 🔹 Inject engineered features
        if features is not None:
            f = self.relu_feat(self.feat_fc2(self.feat_fc1(features)))
            f = f.unsqueeze(-1).expand(-1, f.size(1), out.size(-1))  # match temporal dim
            out = torch.cat([out, f], dim=1)

        out = self.conv_out2(out)  
        return out

    
    def get_layer_internal_result(self, layer, x):
        inter_res = [x]
        for child in layer.children():
            inter_res.append(child(inter_res[-1]))
        return inter_res
