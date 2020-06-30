
class FCN_model(nn.Module):
    def __init__(self,NumClassesOut,N_time,N_Features,N_LSTM_Out=128,N_LSTM_layers = 1
                 ,Conv1_NF = 128,Conv2_NF = 256,Conv3_NF = 128,self.lstmDropP = 0.8,self.FC_DropP = 0.3):
        super(FCN_model,self).__init__()
        
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features,self.N_LSTM_Out,self.N_LSTM_layers)
        self.C1 = nn.Conv1d(self.N_Features,self.Conv1_NF,8)
        self.C2 = nn.Conv1d(self.Conv1_NF,self.Conv2_NF,5)
        self.C3 = nn.Conv1d(self.Conv2_NF,self.Conv3_NF,3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out,self.NumClassesOut)
    
    def init_hidden(self):
        
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        return h0,c0
    
    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        h0,c0 = self.init_hidden()
        x1, (ht,ct) = self.lstm(x, (h0, c0))
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.FC(x_all)
        return x_out
