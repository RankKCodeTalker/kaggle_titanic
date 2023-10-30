import torch


########k折划分############        
def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i: ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    #print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid,y_valid
 
 
def k_fold(k, X_train, y_train, num_epochs=3,learning_rate=0.001, weight_decay=0.1, batch_size=5):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0,0
    
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) # 获取k折交叉验证的训练和验证数据
        net =  Net()  ### 实例化模型
        ### 每份数据进行训练,体现步骤三####
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size) 
       
        print('*'*25,'第',i+1,'折','*'*25)
        print('train_loss:%.6f'%train_ls[-1][0],'train_acc:%.4f\n'%valid_ls[-1][1],\
              'valid loss:%.6f'%valid_ls[-1][0],'valid_acc:%.4f'%valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
    print('#'*10,'最终k折交叉验证结果','#'*10) 
    ####体现步骤四#####
    print('train_loss_sum:%.4f'%(train_loss_sum/k),'train_acc_sum:%.4f\n'%(train_acc_sum/k),\
          'valid_loss_sum:%.4f'%(valid_loss_sum/k),'valid_acc_sum:%.4f'%(valid_acc_sum/k))
