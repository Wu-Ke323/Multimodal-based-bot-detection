from torch import nn
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from layer import RGATLayer
import pytorch_lightning as pl
from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
import torch.nn.functional as F



class RGATDetector(pl.LightningModule):
    def __init__(self, args):
        super(RGATDetector, self).__init__()
        self.edge_index = torch.load(args.path + "edge_index.pt", map_location="cuda")
        self.edge_type = torch.load(args.path + "edge_type.pt", map_location="cuda")
        self.label = torch.load(args.path + "label.pt", map_location="cuda")

        self.lr = args.lr
        self.l2_reg = args.l2_reg

        self.cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cuda")
        self.prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cuda")
        self.tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cuda")
        self.des_features = torch.load(args.path + "des_tensor.pt", map_location="cuda")
        self.image = torch.load(args.path + "user_photo_feature1_8.pt", map_location="cuda")
        self.bg_features = torch.load(args.path + "user_feature_bg_new.pt", map_location="cuda")
        self.avatar_features = torch.load(args.path + "user_feature_avatar_new.pt", map_location="cuda")
        self.video_features = torch.load(args.path + "user_video1_5_feature.pt", map_location="cuda")

        self.in_linear_numeric1 = nn.Linear(args.numeric_num, int(args.linear_channels / 8), bias=True)
        self.in_linear_bool1 = nn.Linear(args.cat_num, int(args.linear_channels / 8), bias=True)
        self.in_linear_tweet1 = nn.Linear(args.tweet_channel, int(args.linear_channels / 8), bias=True)
        self.in_linear_des1 = nn.Linear(args.des_channel, int(args.linear_channels / 8), bias=True)

        self.in_linear_bg = nn.Linear(768, int(args.linear_channels / 8), bias=True)
        self.in_linear_avatar = nn.Linear(768, int(args.linear_channels / 8), bias=True)
        self.in_linear_image = nn.Linear(768, int(args.linear_channels / 8), bias=True)
        self.in_linear_video_feature = nn.Linear(768, int(args.linear_channels/8), bias=True)


        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGAT_layer1 = RGATLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel,
                                   GAT_heads=args.GAT_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGAT_layer2 = RGATLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel,
                                   GAT_heads=args.GAT_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        #self.out2 = torch.nn.Linear(128, 64)
        self.out3 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        device = torch.device('cuda:0')

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        train_batch = train_batch.squeeze(0)

        user_features_numeric1 = self.drop(self.ReLU(self.in_linear_numeric1(self.prop_features)))
        user_features_bool1 = self.drop(self.ReLU(self.in_linear_bool1(self.cat_features)))
        user_features_tweet1 = self.drop(self.ReLU(self.in_linear_tweet1(self.tweet_features)))
        user_features_des1 = self.drop(self.ReLU(self.in_linear_des1(self.des_features)))

        user_features_text1 = torch.cat(
            (user_features_numeric1, user_features_bool1, user_features_tweet1, user_features_des1), dim=1)
        user_features_bg = self.drop(self.ReLU(self.in_linear_bg(self.bg_features)))
        user_features_avatar = self.drop(self.ReLU(self.in_linear_avatar(self.avatar_features)))
        #user_features_bg_avatar = torch.cat((user_features_bg, user_features_avatar), dim=1)
        user_features_video = self.drop(self.ReLU(self.in_linear_video_feature(self.video_features)))
        user_features_image= self.drop(self.ReLU(self.in_linear_image(self.image)))

        user_features = torch.cat((user_features_text1, user_features_bg, user_features_avatar, user_features_video,user_features_image), dim=1)

        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features = self.ReLU(self.RGAT_layer1(user_features, self.edge_index,self.edge_type))
        user_features = self.ReLU(self.RGAT_layer2(user_features, self.edge_index, self.edge_type))

        user_features = self.drop(self.ReLU(self.out1(self.ReLU(user_features))))
        #user_features = self.drop(self.ReLU(self.out2(self.ReLU(user_features))))
        pred = self.out3(user_features[train_batch])
        pred_binary = torch.argmax(pred, dim=1)

        loss = self.CELoss(pred, self.label[train_batch])
        acc = accuracy_score(self.label[train_batch].cpu(), pred_binary.cpu())

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            val_batch = val_batch.squeeze(0)

            user_features_numeric1 = self.drop(self.ReLU(self.in_linear_numeric1(self.prop_features)))
            user_features_bool1 = self.drop(self.ReLU(self.in_linear_bool1(self.cat_features)))
            user_features_tweet1 = self.drop(self.ReLU(self.in_linear_tweet1(self.tweet_features)))
            user_features_des1 = self.drop(self.ReLU(self.in_linear_des1(self.des_features)))

            user_features_text1 = torch.cat(
                (user_features_numeric1, user_features_bool1, user_features_tweet1, user_features_des1), dim=1)
            user_features_bg = self.drop(self.ReLU(self.in_linear_bg(self.bg_features)))
            user_features_avatar = self.drop(self.ReLU(self.in_linear_avatar(self.avatar_features)))
            # user_features_bg_avatar = torch.cat((user_features_bg, user_features_avatar), dim=1)
            user_features_video = self.drop(self.ReLU(self.in_linear_video_feature(self.video_features)))
            user_features_image = self.drop(self.ReLU(self.in_linear_image(self.image)))

            user_features = torch.cat(
                (user_features_text1, user_features_bg, user_features_avatar, user_features_video,user_features_image),
                dim=1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGAT_layer1(user_features, self.edge_index, self.edge_type))
            user_features = self.ReLU(self.RGAT_layer2(user_features, self.edge_index, self.edge_type))

            user_features = self.drop(self.ReLU(self.out1(self.ReLU(user_features))))
            #user_features = self.drop(self.ReLU(self.out2(self.ReLU(user_features))))
            pred = self.out3(user_features[val_batch])
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(self.label[val_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[val_batch].cpu(), pred_binary.cpu())

            self.log("val_acc", acc)
            self.log("val_f1", f1)

            print("acc: {} f1: {}".format(acc, f1))

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            test_batch = test_batch.squeeze(0)
            user_features_numeric1 = self.drop(self.ReLU(self.in_linear_numeric1(self.prop_features)))
            user_features_bool1 = self.drop(self.ReLU(self.in_linear_bool1(self.cat_features)))
            user_features_tweet1 = self.drop(self.ReLU(self.in_linear_tweet1(self.tweet_features)))
            user_features_des1 = self.drop(self.ReLU(self.in_linear_des1(self.des_features)))

            user_features_text1 = torch.cat(
                (user_features_numeric1, user_features_bool1, user_features_tweet1, user_features_des1), dim=1)
            user_features_bg = self.drop(self.ReLU(self.in_linear_bg(self.bg_features)))
            user_features_avatar = self.drop(self.ReLU(self.in_linear_avatar(self.avatar_features)))
            # user_features_bg_avatar = torch.cat((user_features_bg, user_features_avatar), dim=1)
            user_features_video = self.drop(self.ReLU(self.in_linear_video_feature(self.video_features)))
            user_features_image = self.drop(self.ReLU(self.in_linear_image(self.image)))

            user_features = torch.cat(
                (user_features_text1, user_features_bg, user_features_avatar,  user_features_video,user_features_image),
                dim=1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGAT_layer1(user_features, self.edge_index, self.edge_type))
            user_features = self.ReLU(self.RGAT_layer2(user_features, self.edge_index, self.edge_type))

            user_features = self.drop(self.ReLU(self.out1(self.ReLU(user_features))))
            #user_features = self.drop(self.ReLU(self.out2(self.ReLU(user_features))))
            pred = self.out3(user_features[test_batch])
            #torch.save(pred,'tvp_pred_v840.pt')
            #tmp = torch.load('tvp_pred_v840.pt')
            #print(tmp.size())
            pred_binary = torch.argmax(pred, dim=1)
            #torch.save(pred_binary,'tvp_binary_v840.pt')

            acc = accuracy_score(self.label[test_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[test_batch].cpu(), pred_binary.cpu())
            precision = precision_score(self.label[test_batch].cpu(), pred_binary.cpu())
            recall = recall_score(self.label[test_batch].cpu(), pred_binary.cpu())
            auc = roc_auc_score(self.label[test_batch].cpu(), pred[:, 1].cpu())

            self.log("acc", acc)
            self.log("f1", f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(
    description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="./preprocessed/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--GAT_head", type=int, default=2, help="description channel")
parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
parser.add_argument("--batch_size", type=int, default=128, help="description channel")
parser.add_argument("--epochs", type=int, default=30, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        # torch.manual_seed(args.random_seed)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    train_dataset = BotDataset(name="train")
    valid_dataset = BotDataset(name="valid")
    test_dataset = BotDataset(name="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)  # , num_workers=8
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)

    model = RGATDetector(args)
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1,
                         callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, valid_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    #best_path = './lightning_logs/version_840/checkpoints/val_acc=0.8732.ckpt'  #self.in_linear_video_feature改为self.in_linear_video1_feature
    best_model = RGATDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
