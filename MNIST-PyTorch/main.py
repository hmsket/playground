import torch
import torch.nn as nn

import mynet as mn
import dataset as ds
import torch.utils.data as data


def train_model(model, optimizer, loss_module, data_loader, device, epochs=100):

    # モデルを学習モードに設定する
    model.train()

    for _ in range(epochs):
    
        # DataLoderからバッチの取得
        for inputs, labels in data_loader:

            # dataをGPUに移す
            inputs = inputs.to(device)
            labels = labels.to(device)

            # バッチをモデルへ入力し、出力値を計算する
            outputs = model(inputs)
            outputs = outputs.squeeze(dim=1) # [BATCH_SIZE, 1] -> [BATCH_SIZE] に変換

            # 出力値と教師値から損失を計算する
            loss = loss_module(outputs, labels.float())

            # 損失をもとに各パラメータの勾配を計算する
            optimizer.zero_grad()
            loss.backward()
            
            # パラメータを更新する
            optimizer.step()


def eval_model(model, data_loader, device):
    
    # モデルを推論モードにする
    model.eval()

    true_preds, num_preds = 0., 0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= 0.5).long()
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]
    
    acc = true_preds / num_preds
    print(f"正解率：{100.0*acc:4.2f}%")


def main():

    """ 準備
    """
    # モデルの定義
    model = mn.MyNet(ni=2, nh=4, no=1)
    
    # DataLoaderの準備
    dataset = ds.XORDataset(size=2500)
    data_loader = data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    
    # 損失関数の定義
    loss_module = nn.BCEWithLogitsLoss()

    # optimizerの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # モデルをGPUに移す
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    """ 学習
    """
    train_model(model, optimizer, loss_module, data_loader, device, epochs=100)

    """ モデルの保存
    """
    state_dict = model.state_dict()
    torch.save(state_dict, "model.tar")

    """ 推論
    """
    eval_model(model, data_loader, device)

if __name__ == "__main__":
    main()
